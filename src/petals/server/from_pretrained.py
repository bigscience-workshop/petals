"""
Utils for fetching pretrained model parts. Currently, this relies on huggingface transformers' from_pretrained code.
If necessary, one can rewrite this to implement a different behavior, such as:
 - loading files from a local data source (e.g. S3)
 - load files via BitTorrent ( https://pypi.org/project/libtorrent/ ) or IPFS( https://docs.ipfs.io/how-to )
 - fetch the weights over IPoAC, using a fleet of trained pigeons ( http://www.faqs.org/rfcs/rfc1149.html )

"""
import itertools
import json
import time
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from hivemind.utils.logging import get_logger
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from transformers import AutoConfig, PretrainedConfig
from transformers.modeling_utils import WEIGHTS_NAME
from transformers.utils import get_file_from_repo

from petals.server.block_utils import get_block_size
from petals.server.model_specs import MODEL_SPECS
from petals.utils.disk_cache import DEFAULT_CACHE_DIR, allow_cache_reads, allow_cache_writes, free_disk_space_for

logger = get_logger(__name__)


class AutoBlockConfig:
    @staticmethod
    def from_pretrained(*args, **kwargs) -> PretrainedConfig:
        config = AutoConfig.from_pretrained(*args, **kwargs)

        if config.model_type not in MODEL_SPECS:
            raise ValueError(f"Unsupported model architecture: {config.model_type}")
        model_spec = MODEL_SPECS[config.model_type]

        config.block_class = model_spec.block_class
        config.attn_class = model_spec.attn_class
        config.block_prefix = model_spec.block_prefix

        for dest, src in model_spec.config_map.items():
            setattr(config, dest, getattr(config, src))

        return config


def load_pretrained_block(
    model_name: str,
    block_index: int,
    config: Optional[PretrainedConfig] = None,
    torch_dtype: Union[torch.dtype, str] = "auto",
    use_auth_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_disk_space: Optional[int] = None,
) -> nn.Module:
    assert torch_dtype in DTYPE_MAP.values(), f"torch_dtype must be one of {list(DTYPE_MAP.values())}"

    if config is None:
        config = AutoBlockConfig.from_pretrained(model_name, use_auth_token=use_auth_token)
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    with init_empty_weights():
        block = config.block_class(config)

    block_prefix = f"{config.block_prefix}.{block_index}."
    state_dict = _load_state_dict_from_repo(
        model_name,
        block_prefix,
        config,
        use_auth_token=use_auth_token,
        cache_dir=cache_dir,
        max_disk_space=max_disk_space,
    )
    state_dict = {
        param_name[len(block_prefix) :]: param
        for param_name, param in state_dict.items()
        if param_name.startswith(block_prefix)
    }

    # dummy load, check that keys match
    report = block.load_state_dict(state_dict, strict=True)
    assert not report.missing_keys, f"Some block weights are missing: {report.missing_keys}"

    for param_name, _ in block.named_parameters():
        assert param_name in state_dict, f"{param_name} not in state dict"
        param = state_dict[param_name]
        if torch_dtype != "auto" and not str(param.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            param = param.to(torch_dtype)
        set_module_tensor_to_device(block, param_name, "cpu", value=param, dtype=param.dtype)

    logger.info(f"Loaded {model_name} block {block_index}, {report}")
    return block


StateDict = Dict[str, torch.Tensor]


def _load_state_dict_from_repo(
    model_name: str,
    block_prefix: str,
    config: PretrainedConfig,
    *,
    use_auth_token: Optional[str] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
    min_backoff: float = 5,
) -> StateDict:
    index_file = get_file_from_repo(
        model_name, filename="pytorch_model.bin.index.json", use_auth_token=use_auth_token, cache_dir=cache_dir
    )
    with open(index_file) as f:
        index = json.load(f)

    filenames = {
        filename for param_name, filename in index["weight_map"].items() if param_name.startswith(block_prefix)
    }
    if len(filenames) > 1:
        raise RuntimeError(
            f"Block {block_prefix}* is stored in {filenames}, but Petals can't load blocks divided into multiple files yet"
        )
    [filename] = filenames
    logger.debug(f"Loading {block_prefix}* from {filename}")

    # First, try to find the weights locally
    try:
        with allow_cache_reads(cache_dir):
            return _load_state_dict_from_file(
                model_name, filename, use_auth_token=use_auth_token, cache_dir=cache_dir, local_files_only=True
            )
    except Exception:
        logger.debug(
            f"Failed to load block {block_prefix}* from cache, proceeding to downloading the block", exc_info=True
        )

    # If not found, ensure that we have enough disk space to download them (maybe remove something)
    for attempt_no in itertools.count():
        try:
            with allow_cache_writes(cache_dir):
                url = hf_hub_url(model_name, filename)
                file_size = get_hf_file_metadata(url, token=use_auth_token).size
                gib = 1024**3
                logger.debug(f"Shard size for {filename}: {file_size / gib:.2f} GiB")

                free_disk_space_for(model_name, file_size, cache_dir=cache_dir, max_disk_space=max_disk_space)

                return _load_state_dict_from_file(
                    model_name, filename, use_auth_token=use_auth_token, cache_dir=cache_dir, local_files_only=False
                )
        except Exception as e:
            delay = min_backoff * (2**attempt_no)
            logger.warning(
                f"Failed to load block {block_prefix}* from HF Hub (retry in {delay:.0f} sec)", exc_info=True
            )
            time.sleep(delay)


def _load_state_dict_from_file(
    model_name: str, filename: str, *, use_auth_token: Optional[str], cache_dir: str, local_files_only: bool
) -> StateDict:
    path = get_file_from_repo(
        model_name,
        filename=filename,
        use_auth_token=use_auth_token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if path is None:
        raise RuntimeError(f"Failed to load file {filename} from repo {model_name}")
    return torch.load(path, map_location="cpu")


DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")
