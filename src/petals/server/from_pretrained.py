"""
Utils for fetching pretrained model parts. Currently, this relies on huggingface transformers' from_pretrained code.
If necessary, one can rewrite this to implement a different behavior, such as:
 - loading files from a local data source (e.g. S3)
 - load files via BitTorrent ( https://pypi.org/project/libtorrent/ ) or IPFS( https://docs.ipfs.io/how-to )
 - fetch the weights over IPoAC, using a fleet of trained pigeons ( http://www.faqs.org/rfcs/rfc1149.html )

"""
import json
import time
from contextlib import suppress
from typing import Dict, Optional, Union

import safetensors
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from hivemind.utils.logging import get_logger
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import get_file_from_repo
from transformers.quantizers import AutoHfQuantizer

from petals.constants import DTYPE_MAP
from petals.models.mixtral import WrappedMixtralBlock
from petals.server.block_utils import get_model_block, resolve_block_dtype
from petals.utils.auto_config import AutoDistributedConfig
from petals.utils.disk_cache import DEFAULT_CACHE_DIR, allow_cache_reads, allow_cache_writes, free_disk_space_for
from petals.utils.hf_auth import always_needs_auth
from petals.utils.convert_block import is_gptq_quant

logger = get_logger(__name__)


def load_pretrained_block(
    model_name: str,
    block_index: int,
    *,
    config: Optional[PretrainedConfig] = None,
    torch_dtype: Union[torch.dtype, str] = "auto",
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: Optional[str] = None,
    max_disk_space: Optional[int] = None,
) -> nn.Module:
    if config is None:
        config = AutoDistributedConfig.from_pretrained(model_name, use_auth_token=token)
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    assert torch_dtype in DTYPE_MAP.values(), f"torch_dtype must be one of {list(DTYPE_MAP.values())}"
    torch_dtype = resolve_block_dtype(config, torch_dtype)
    print("block_index",block_index)
    with init_empty_weights():
        block = get_model_block(config, layer_idx=block_index)

    block_prefix = f"{config.block_prefix}.{block_index}."
    # print(config)
    print(is_gptq_quant(config))
    if is_gptq_quant(config):
        print("Now loading GPTQ")
        hf_quantizer = AutoHfQuantizer.from_config(config.quantization_config, pre_quantized=True)
        hf_quantizer.optimum_quantizer.block_name_to_quantize = str(block_index)
        tmp_block_list = torch.nn.ModuleList([block])
        tmp_block_list.__class__.main_input_name = "input_ids"
        torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
        device_map = hf_quantizer.update_device_map("cpu")
        hf_quantizer.preprocess_model(
            model=tmp_block_list, device_map=device_map, keep_in_fp32_modules=False,
        )
        # if model_name[-1] != '.':
        #     model_name += '.'

    state_dict = _load_state_dict_from_repo(
        model_name,
        block_prefix,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        max_disk_space=max_disk_space,
    )
    print(model_name, block_prefix, revision,token,cache_dir,max_disk_space)

    # print("now printing", state_dict)
    # print("block.named_parameters()",block.named_parameters())
    if is_gptq_quant(config):
        print("loading state_dict")
        block.load_state_dict(state_dict, assign=True, strict=False)
    else:
        for param_name, _ in block.named_parameters():
            print(param_name)
            assert param_name in state_dict, f"{param_name} not in state dict"
            param = state_dict[param_name]
            if not str(param.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
                param = param.to(torch_dtype)
            set_module_tensor_to_device(block, param_name, "cpu", value=param, dtype=param.dtype)

    logger.info(f"Loaded {model_name} block {block_index}")
    print(block.self_attn.q_proj.qzeros)
    return block


StateDict = Dict[str, torch.Tensor]


def _load_state_dict_from_repo(
    model_name: str,
    block_prefix: str,
    *,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
) -> StateDict:
    if always_needs_auth(model_name) and token is None:
        token = True

    index_file = _find_index_file(model_name, revision=revision, token=token, cache_dir=cache_dir)
    if index_file.endswith(".index.json"):  # Sharded model
        path = get_file_from_repo(model_name, filename=index_file, use_auth_token=token, cache_dir=cache_dir)
        if path is None:
            # _find_index_file() told that a file exists but we can't get it (e.g., it just disappeared)
            raise ValueError(f"Failed to get file {index_file}")

        with open(path) as f:
            index = json.load(f)
        filenames = {
            filename for param_name, filename in index["weight_map"].items() if param_name.startswith(block_prefix)
        }
        if not filenames:
            raise RuntimeError(f"Block {block_prefix}* not found in the index: {index['weight_map']}")
    else:  # Non-sharded model
        filenames = {index_file}
    logger.debug(f"Loading {block_prefix}* from {filenames}")

    state_dict = {}
    for filename in filenames:
        shard_state_dict = _load_state_dict_from_repo_file(
            model_name,
            filename,
            block_prefix=block_prefix,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            max_disk_space=max_disk_space,
        )
        shard_state_dict = {
            param_name[len(block_prefix) :]: param
            for param_name, param in shard_state_dict.items()
            if param_name.startswith(block_prefix)
        }  # Remove unused parameters from memory
        state_dict.update(shard_state_dict)
    return state_dict


INDEX_FILES = ["model.safetensors.index.json", "model.safetensors", "pytorch_model.bin.index.json", "pytorch_model.bin"]


def _find_index_file(
    model_name: str, *, revision: Optional[str] = None, token: Optional[Union[str, bool]] = None, cache_dir: str
) -> str:
    # If we have cached weights (e.g., Pickle from older Petals versions), reuse them
    for filename in INDEX_FILES:
        path = get_file_from_repo(
            model_name,
            filename,
            revision=revision,
            use_auth_token=token,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        if path is not None:
            return filename

    # If we don't, prefer Safetensors when possible
    # (we don't download files here since we can't account for max_disk_space in case of large files)
    for filename in INDEX_FILES:
        with suppress(EntryNotFoundError):
            get_hf_file_metadata(hf_hub_url(model_name, filename, revision=revision), token=token)
            return filename

    raise ValueError(
        f"Repo {model_name} does not contain weights in a supported format: files {INDEX_FILES} do not exist"
    )


def _load_state_dict_from_repo_file(
    model_name: str,
    filename: str,
    *,
    block_prefix: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
    delay: float = 30,
) -> StateDict:
    # First, try to find the weights locally
    try:
        with allow_cache_reads(cache_dir):
            path = get_file_from_repo(
                model_name,
                filename,
                revision=revision,
                use_auth_token=token,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            if path is not None:
                return _load_state_dict_from_local_file(path, block_prefix=block_prefix)
    except Exception:
        logger.warning(f"Cache for file {filename} is corrupted, it will be downloaded again", exc_info=True)

    # If not found, ensure that we have enough disk space to download them (maybe remove something)
    while True:
        try:
            with allow_cache_writes(cache_dir):
                url = hf_hub_url(model_name, filename, revision=revision)
                file_size = get_hf_file_metadata(url, token=token).size
                if file_size is not None:
                    free_disk_space_for(file_size, cache_dir=cache_dir, max_disk_space=max_disk_space)
                else:
                    logger.warning(f"Failed to fetch size of file {filename} from repo {model_name}")

                path = get_file_from_repo(
                    model_name,
                    filename,
                    revision=revision,
                    use_auth_token=token,
                    cache_dir=cache_dir,
                    local_files_only=False,
                )
                if path is None:
                    raise RuntimeError(f"File {filename} does not exist in repo {model_name}")
                return _load_state_dict_from_local_file(path, block_prefix=block_prefix)
        except Exception as e:
            logger.warning(f"Failed to load file {filename} from HF Hub (retry in {delay:.0f} sec)", exc_info=True)
            time.sleep(delay)


def _load_state_dict_from_local_file(path: str, *, block_prefix: Optional[str] = None) -> StateDict:
    if path.endswith(".bin"):
        return torch.load(path, map_location="cpu")

    if path.endswith(".safetensors"):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys() if block_prefix is None or key.startswith(block_prefix)}

    raise ValueError(f"Unknown weight format: {path}")
