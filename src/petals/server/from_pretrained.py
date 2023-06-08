"""
Utils for fetching pretrained model parts. Currently, this relies on huggingface transformers' from_pretrained code.
If necessary, one can rewrite this to implement a different behavior, such as:
 - loading files from a local data source (e.g. S3)
 - load files via BitTorrent ( https://pypi.org/project/libtorrent/ ) or IPFS( https://docs.ipfs.io/how-to )
 - fetch the weights over IPoAC, using a fleet of trained pigeons ( http://www.faqs.org/rfcs/rfc1149.html )

"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Optional, OrderedDict, Union

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from hivemind.utils.logging import get_logger
from transformers import AutoConfig, PretrainedConfig
from transformers.modeling_utils import WEIGHTS_NAME
from transformers.models.bloom.modeling_bloom import BloomAttention
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.utils import get_file_from_repo

from petals.bloom.block import WrappedBloomBlock
from petals.llama.block import WrappedLlamaBlock
from petals.server.block_utils import get_block_size
from petals.utils.disk_cache import DEFAULT_CACHE_DIR, allow_cache_reads, allow_cache_writes, free_disk_space_for

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    block_class: Type[nn.Module]
    attn_class: Type[nn.Module]  # An nn.Module with attention, expected to have the .num_heads attribute
    block_prefix: str
    config_map: Dict[str, str] = field(
        default_factory=dict
    )  # How to translate config keys into the bigscience/bloom format


MODEL_SPECS = {
    "bloom": ModelSpec(block_class=WrappedBloomBlock, attn_class=BloomAttention, block_prefix="h"),
    "llama": ModelSpec(
        block_class=WrappedLlamaBlock,
        attn_class=LlamaAttention,
        block_prefix="model.layers",
        config_map={
            "n_head": "num_attention_heads",
            "n_layer": "num_hidden_layers",
        },
    ),
}


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
) -> WrappedLlamaBlock:
    assert torch_dtype in DTYPE_MAP.values(), f"torch_dtype must be one of {list(DTYPE_MAP.values())}"

    if config is None:
        config = AutoBlockConfig.from_pretrained(model_name, use_auth_token=use_auth_token)
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    with init_empty_weights():
        block = config.block_class(config)

    block_prefix = f"{config.block_prefix}.{block_index}."
    state_dict = _load_state_dict(
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


def _load_state_dict(
    model_name: str,
    block_prefix: str,
    config: PretrainedConfig,
    *,
    use_auth_token: Optional[str] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
    min_backoff: float = 5,
) -> OrderedDict[str, torch.Tensor]:
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
            archive_file = get_file_from_repo(
                model_name,
                filename=filename,
                use_auth_token=use_auth_token,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            if archive_file is not None:
                return torch.load(archive_file, map_location="cpu")
    except Exception:
        logger.debug(
            f"Failed to load block {block_prefix}* from cache. The block will be downloaded again", exc_info=True
        )

    # If not found, ensure that we have enough disk space to download them (maybe remove something)
    for attempt_no in itertools.count():
        try:
            with allow_cache_writes(cache_dir):
                # block_size = get_block_size(config, "disk")
                # # FIXME: Get correct file size
                # free_disk_space_for(
                #     model_name, block_size, cache_dir=cache_dir, max_disk_space=max_disk_space
                # )

                archive_file = get_file_from_repo(
                    model_name,
                    filename=filename,
                    use_auth_token=use_auth_token,
                    cache_dir=cache_dir,
                    local_files_only=False,
                )
                return torch.load(archive_file, map_location="cpu")
        except Exception as e:
            delay = min_backoff * (2**attempt_no)
            logger.warning(
                f"Failed to load block {block_prefix}* from HF Hub (retry in {delay:.0f} sec)", exc_info=True
            )
            time.sleep(delay)


DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")
