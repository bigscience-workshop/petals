import re
import time
from typing import List, Optional

import bitsandbytes as bnb
import torch.nn as nn
from hivemind.utils.logging import get_logger
from huggingface_hub import HfFileSystem, get_hf_file_metadata, hf_hub_url
from peft.tuners import lora
from peft.utils import COMMON_LAYERS_PATTERN, CONFIG_NAME, SAFETENSORS_WEIGHTS_NAME, PeftConfig
from safetensors import safe_open
from safetensors.torch import load_file
from transformers.utils import get_file_from_repo

from petals.utils.disk_cache import allow_cache_reads, allow_cache_writes, free_disk_space_for
from petals.utils.misc import QuantType

logger = get_logger(__name__)


def check_peft_repository(repo_id: str) -> bool:
    fs = HfFileSystem()
    list_of_files = fs.glob(f"{repo_id}/{SAFETENSORS_WEIGHTS_NAME}", detail=False)
    return len(list_of_files) > 0


def load_specific_module(block_idx: int, filepath: str, framework: str = "pt", device: Optional[int] = None):
    tensors = dict()
    is_tensors_found = dict()
    common_layer_patter_re = (
        ".+\." + "".join(f"({common_name})?" for common_name in COMMON_LAYERS_PATTERN) + f"\.({block_idx})?\..+"
    )
    with safe_open(filepath, framework=framework, device=device) as f:
        for k in f.keys():
            if re.match(common_layer_patter_re, k):
                is_tensors_found[block_idx] = True
                tensors[k] = f.get_tensor(k)
        if not is_tensors_found.get(block_idx, False):
            logger.warning(f"There is no peft weights for block {block_idx}")
        return tensors


def get_adapter_from_repo(repo_id: str, block_idx: Optional[int] = None, device: Optional[int] = None, **kwargs):
    config_path = get_file_from_repo(repo_id, CONFIG_NAME, **kwargs)
    if config_path is None:
        raise RuntimeError(f"File {CONFIG_NAME} does not exist in repo {repo_id}")
    config = PeftConfig.from_json_file(config_path)

    weight_path = get_file_from_repo(repo_id, SAFETENSORS_WEIGHTS_NAME, **kwargs)
    if weight_path is None:
        raise RuntimeError(f"File {SAFETENSORS_WEIGHTS_NAME} does not exist in repo {repo_id}")
    if block_idx is None:
        return config, load_file(weight_path)
    return config, load_specific_module(block_idx, weight_path, device=device)


def load_peft(
    repo_id: str,
    block_idx: Optional[int] = None,
    device: Optional[int] = None,
    *,
    revision: Optional[str] = None,
    use_auth_token: Optional[str] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
    delay: float = 30,
):
    # TODO: Check is it possible to add safetensors loading inside petals/server/from_pretrained.py and reuse it here

    if not check_peft_repository(repo_id):
        raise ValueError(f"Repo: {repo_id} doesn't have safetensors inside for a safe loading.")

    try:
        with allow_cache_reads(cache_dir):
            return get_adapter_from_repo(
                repo_id,
                block_idx,
                device,
                revision=revision,
                use_auth_token=use_auth_token,
                cache_dir=cache_dir,
                local_files_only=False,
            )
    except Exception:
        logger.warning(f"Cache for peft weights {repo_id} is corrupted, it will be downloaded again", exc_info=True)

    while True:
        try:
            with allow_cache_writes(cache_dir):
                config_url = hf_hub_url(repo_id, CONFIG_NAME, revision=revision)
                config_file_size = get_hf_file_metadata(config_url, token=use_auth_token).size
                weight_url = hf_hub_url(repo_id, SAFETENSORS_WEIGHTS_NAME, revision=revision)
                weight_file_size = get_hf_file_metadata(weight_url, token=use_auth_token).size

                file_size = config_file_size + weight_file_size
                if file_size is not None:
                    free_disk_space_for(file_size, cache_dir=cache_dir, max_disk_space=max_disk_space)
                else:
                    logger.warning(f"Failed to fetch size from peft repo {repo_id}")

                return get_adapter_from_repo(
                    repo_id,
                    block_idx,
                    device,
                    revision=revision,
                    use_auth_token=use_auth_token,
                    cache_dir=cache_dir,
                    local_files_only=False,
                )
        except Exception as e:
            logger.warning(
                f"Failed to load peft weights {repo_id} from HF Hub (retry in {delay:.0f} sec)", exc_info=True
            )
            time.sleep(delay)


def create_lora_adapter(block, quant_type: QuantType):
    for name, module in block.named_modules():
        for child_name, child in module.named_children():
            lora_wrapped_child = None
            if not isinstance(child, (nn.Linear, bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)):
                continue
            if quant_type == QuantType.INT8:
                kwargs = {
                    "has_fp16_weights": False,
                    "threshold": 6.0,
                    "bias": hasattr(child, "bias") and child.bias is not None,
                }
                lora_wrapped_child = lora.Linear8bitLt(
                    child_name,
                    child.in_features,
                    child.out_features,
                    **kwargs,
                )
            elif quant_type == QuantType.NF4:
                kwargs = {
                    "compress_statistics": True,
                    "quant_type": "nf4",
                    "blocksize": 64,
                    "bias": hasattr(child, "bias") and child.bias is not None,
                }
                lora_wrapped_child = lora.Linear4bit(
                    child_name,
                    child.in_features,
                    child.out_features,
                    **kwargs,
                )
            else:
                bias = hasattr(child, "bias") and child.bias is not None
                lora_wrapped_child = lora.Linear(
                    child_name,
                    child.in_features,
                    child.out_features,
                    bias=bias,
                )
            if lora_wrapped_child:
                lora_wrapped_child.active_adapter = None
                lora_wrapped_child.weight = child.weight
                lora_wrapped_child.bias = child.bias
                for p in lora_wrapped_child.parameters():
                    p.requires_grad = False
                setattr(module, child_name, lora_wrapped_child)


def add_adapter_to_block(block, block_index, adapter_name, peft_config, peft_state_dict):
    assert peft_config["peft_type"] == "LORA", "Petals works only with LORA adapters"
    for name, module in block.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, (lora.Linear, lora.Linear8bitLt, lora.Linear4bit)):
                continue

            if child_name in peft_config["target_modules"] or (
                isinstance(peft_config["target_modules"], str)
                and re.fullmatch(peft_config["target_modules"], child_name)
            ):
                is_lora_a_loaded = False
                is_lora_b_loaded = False
                for peft_key in peft_state_dict:
                    if peft_key.find(child_name) == -1:
                        continue

                    if adapter_name not in child.lora_A:
                        child.update_layer(
                            adapter_name,
                            peft_config["r"],
                            peft_config["lora_alpha"],
                            lora_dropout=peft_config["lora_dropout"],
                            init_lora_weights=peft_config["init_lora_weights"],
                        )
                        child.train(False)
                        if peft_config["lora_dropout"] > 0:
                            logger.warning("Loading LoRA config with dropout enabled; this server will disable dropout")
                        for p in child.parameters():
                            p.requires_grad = False

                    if peft_key.endswith(".lora_A.weight"):
                        child.lora_A[adapter_name].weight.data = peft_state_dict[peft_key]
                        is_lora_a_loaded = True
                    elif peft_key.endswith(".lora_A.bias"):
                        raise NotImplementedError(f"LoRA adapters with bias not supported: {peft_key}")
                    elif peft_key.endswith(".lora_B.weight"):
                        child.lora_B[adapter_name].weight.data = peft_state_dict[peft_key]
                        is_lora_b_loaded = True
                    elif peft_key.endswith(".lora_B.bias"):
                        raise NotImplementedError(f"LoRA adapters with bias not supported: {peft_key}")

                if is_lora_a_loaded and is_lora_b_loaded:
                    logger.info(f"Loading {adapter_name} for block {block_index}.{child_name} is ended successfully")
