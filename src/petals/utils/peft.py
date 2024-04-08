import contextlib
import re
import time
from typing import Optional, Sequence, Union

import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from accelerate import init_empty_weights
from hivemind.utils.logging import get_logger
from huggingface_hub import HfFileSystem, get_hf_file_metadata, hf_hub_url
from peft.config import PeftConfig
from peft.tuners import lora
from peft.utils import COMMON_LAYERS_PATTERN, CONFIG_NAME, SAFETENSORS_WEIGHTS_NAME
from safetensors import safe_open
from safetensors.torch import load_file
from transformers.utils import get_file_from_repo

from petals.server.block_utils import get_model_block, resolve_block_dtype
from petals.utils.convert_block import QuantType
from petals.utils.disk_cache import allow_cache_reads, allow_cache_writes, free_disk_space_for
from petals.utils.misc import get_size_in_bytes

logger = get_logger(__name__)


def check_peft_repository(repo_id: str) -> bool:
    return HfFileSystem().exists(f"{repo_id}/{SAFETENSORS_WEIGHTS_NAME}")


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


def get_adapter_from_repo(
    repo_id: str,
    block_idx: Optional[int] = None,
    device: Optional[int] = None,
    *,
    token: Optional[Union[str, bool]] = None,
    **kwargs,
):
    config_path = get_file_from_repo(repo_id, CONFIG_NAME, use_auth_token=token, **kwargs)
    if config_path is None:
        raise RuntimeError(f"File {CONFIG_NAME} does not exist in repo {repo_id}")
    config = PeftConfig.from_json_file(config_path)

    weight_path = get_file_from_repo(repo_id, SAFETENSORS_WEIGHTS_NAME, use_auth_token=token, **kwargs)
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
    token: Optional[Union[str, bool]] = None,
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
                token=token,
                cache_dir=cache_dir,
                local_files_only=False,
            )
    except Exception:
        logger.warning(f"Cache for peft weights {repo_id} is corrupted, it will be downloaded again", exc_info=True)

    while True:
        try:
            with allow_cache_writes(cache_dir):
                config_url = hf_hub_url(repo_id, CONFIG_NAME, revision=revision)
                config_file_size = get_hf_file_metadata(config_url, token=token).size
                weight_url = hf_hub_url(repo_id, SAFETENSORS_WEIGHTS_NAME, revision=revision)
                weight_file_size = get_hf_file_metadata(weight_url, token=token).size

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
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=False,
                )
        except Exception as e:
            logger.warning(
                f"Failed to load peft weights {repo_id} from HF Hub (retry in {delay:.0f} sec)", exc_info=True
            )
            time.sleep(delay)


class AdapterContextMixin:
    """A mixin that makes LoRA-wrapped linear layers obey an adapter set from context"""

    ADAPTER_NOT_SET = "__ADAPTER_NOT_SET"
    _context_active_adapter = ADAPTER_NOT_SET

    @staticmethod
    @contextlib.contextmanager
    def using_adapter(active_adapter: Optional[str]):
        prev, AdapterContextMixin._context_active_adapter = AdapterContextMixin._context_active_adapter, active_adapter
        try:
            yield
        finally:
            AdapterContextMixin._context_active_adapter = prev

    @property
    def active_adapter(self):
        if self._context_active_adapter == self.ADAPTER_NOT_SET:
            logger.warning(f"Layer {self} was called without using_adapter. This should only be used for debug")
        return self._context_active_adapter

    @active_adapter.setter
    def active_adapter(self, value: Optional[str]):
        assert value == self.ADAPTER_NOT_SET, "active adapter can only be changed via .using_adapter" ""


using_adapter = AdapterContextMixin.using_adapter


class LoraLinear(AdapterContextMixin, lora.Linear):
    """LoRA linear layer that uses adapter selected via using_adapter"""


class LoraLinear8bitLt(AdapterContextMixin, lora.Linear8bitLt):
    """LoRA linear 8-bit with outliers that uses adapter selected via using_adapter"""


class LoraLinear4bit(AdapterContextMixin, lora.Linear4bit):
    """LoRA linear 4-bit that uses adapter selected via using_adapter"""


def create_lora_adapter(block, quant_type: QuantType):
    for _, module in block.named_modules():
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
                lora_wrapped_child = LoraLinear8bitLt(
                    AdapterContextMixin.ADAPTER_NOT_SET,
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
                lora_wrapped_child = LoraLinear4bit(
                    AdapterContextMixin.ADAPTER_NOT_SET,
                    child.in_features,
                    child.out_features,
                    **kwargs,
                )
                lora_wrapped_child.compute_dtype = child.compute_dtype
            else:
                bias = hasattr(child, "bias") and child.bias is not None
                lora_wrapped_child = LoraLinear(
                    AdapterContextMixin.ADAPTER_NOT_SET,
                    child.in_features,
                    child.out_features,
                    bias=bias,
                )
            if lora_wrapped_child:
                lora_wrapped_child.weight = child.weight
                lora_wrapped_child.bias = child.bias
                for p in lora_wrapped_child.parameters():
                    p.requires_grad = False
                setattr(module, child_name, lora_wrapped_child)


def add_adapter_to_block(block, block_index, adapter_name, peft_config, peft_state_dict):
    assert peft_config["peft_type"] == "LORA", "Petals works only with LORA adapters"
    if peft_config["lora_dropout"] > 0:
        logger.info(f"Adapter {adapter_name} has dropout enabled, this server will disable dropout")

    for _, module in block.named_modules():
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
                    if child_name not in peft_key:
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
                        for p in child.parameters():
                            p.requires_grad = False

                    if peft_key.endswith(".lora_A.weight"):
                        child.lora_A[adapter_name].weight[...] = peft_state_dict[peft_key]
                        is_lora_a_loaded = True
                    elif peft_key.endswith(".lora_A.bias"):
                        raise NotImplementedError(f"LoRA adapters with bias not supported: {peft_key}")
                    elif peft_key.endswith(".lora_B.weight"):
                        child.lora_B[adapter_name].weight[...] = peft_state_dict[peft_key]
                        is_lora_b_loaded = True
                    elif peft_key.endswith(".lora_B.bias"):
                        raise NotImplementedError(f"LoRA adapters with bias not supported: {peft_key}")

                if is_lora_a_loaded and is_lora_b_loaded:
                    logger.debug(f"Loaded adapter {adapter_name} for block {block_index}.{child_name}")
                elif is_lora_a_loaded or is_lora_b_loaded:
                    raise ValueError(f"Invalid adapter {adapter_name} for block {block_index}.{child_name}")
    logger.info(f"Loaded adapter {adapter_name} for block {block_index}")


def estimate_adapter_memory_per_block(
    block_config: transformers.PretrainedConfig,
    torch_dtype: Optional[torch.dtype],
    adapters: Sequence[str],
    **load_peft_kwargs,
) -> int:
    """Get the number of extra bytes used to store a set of adapters per given block"""
    with init_empty_weights(include_buffers=True):
        block = get_model_block(block_config)
        base_block_parameters = sum(p.numel() for p in block.parameters())
        create_lora_adapter(block, quant_type=QuantType.NONE)

        for adapter in adapters:
            peft_config, peft_state_dict = load_peft(adapter, block_idx=0, **load_peft_kwargs)
            assert peft_config["peft_type"].upper() == "LORA", "only LoRA adapters are supported for now"
            add_adapter_to_block(
                block, block_index=0, adapter_name=adapter, peft_config=peft_config, peft_state_dict=peft_state_dict
            )
        adapter_parameters = sum(p.numel() for p in block.parameters()) - base_block_parameters
    bytes_per_parameter = get_size_in_bytes(resolve_block_dtype(block_config, torch_dtype))
    return adapter_parameters * bytes_per_parameter
