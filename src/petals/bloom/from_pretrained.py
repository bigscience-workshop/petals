"""
Utils for fetching pretrained model parts. Currently, this relies on huggingface transformers' from_pretrained code.
If necessary, one can rewrite this to implement a different behavior, such as:
 - loading files from a local data source (e.g. S3)
 - load files via BitTorrent ( https://pypi.org/project/libtorrent/ ) or IPFS( https://docs.ipfs.io/how-to )
 - fetch the weights over IPoAC, using a fleet of trained pigeons ( http://www.faqs.org/rfcs/rfc1149.html )

"""
from __future__ import annotations

from typing import Optional, OrderedDict, Union

import torch
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from transformers.modeling_utils import WEIGHTS_NAME
from transformers.models.bloom.configuration_bloom import BloomConfig
from transformers.utils import get_file_from_repo

from petals.bloom.block import WrappedBloomBlock
from petals.server.block_utils import get_block_size
from petals.utils.disk_cache import DEFAULT_CACHE_DIR, allow_cache_reads, allow_cache_writes, free_disk_space_for

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)

CLIENT_BRANCH = "main"
BLOCK_BRANCH_PREFIX = "block_"
USER_AGENT = {"file_type": "model", "framework": "pytorch", "from_auto_class": False}
FORCE_DOWNLOAD = False
RESUME_DOWNLOAD = False
LOCAL_FILES_ONLY = False


def load_pretrained_block(
    converted_model_name_or_path: str,
    block_index: int,
    config: Optional[BloomConfig] = None,
    torch_dtype: Union[torch.dtype, str] = "auto",
    use_auth_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_disk_space: Optional[int] = None,
) -> WrappedBloomBlock:
    """Load one BloomBlock from a converted model. See convert_model.py (or README.md) on how to convert it."""

    if config is None:
        config = BloomConfig.from_pretrained(converted_model_name_or_path, use_auth_token=use_auth_token)
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    block = WrappedBloomBlock(config)
    state_dict = _load_state_dict(
        converted_model_name_or_path,
        block_index,
        config,
        use_auth_token=use_auth_token,
        cache_dir=cache_dir,
        max_disk_space=max_disk_space,
    )

    if torch_dtype == "auto":
        with torch.no_grad():
            for name, param in block.named_parameters():
                assert name in state_dict, f"{name} not in state dict"
                param.data = param.data.to(state_dict[name].dtype)
    else:
        assert torch_dtype in DTYPE_MAP.values(), f"torch_dtype must be one of {list(DTYPE_MAP.values())}"
        block = block.to(dtype=torch_dtype)

    report = block.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded {converted_model_name_or_path} block {block_index}, {report}")
    return block


def _load_state_dict(
    pretrained_model_name_or_path: str,
    block_index: int,
    config: BloomConfig,
    *,
    use_auth_token: Optional[str] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
) -> OrderedDict[str, torch.Tensor]:
    revision = BLOCK_BRANCH_PREFIX + str(block_index)

    # First, try to find the weights locally
    with allow_cache_reads(cache_dir):
        try:
            archive_file = get_file_from_repo(
                pretrained_model_name_or_path,
                filename=WEIGHTS_NAME,
                revision=revision,
                use_auth_token=use_auth_token,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            if archive_file is not None:
                return torch.load(archive_file, map_location="cpu")
        except Exception as e:
            logger.info(f"Failed to load block {block_index} from cache: {e}. The block will be downloaded again")

    # If not found, ensure that we have enough disk space to download them (maybe remove something)
    with allow_cache_writes(cache_dir):
        block_size = get_block_size(config, "disk")
        free_disk_space_for(
            pretrained_model_name_or_path, block_size, cache_dir=cache_dir, max_disk_space=max_disk_space
        )

        archive_file = get_file_from_repo(
            pretrained_model_name_or_path,
            filename=WEIGHTS_NAME,
            revision=revision,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            local_files_only=False,
        )
        return torch.load(archive_file, map_location="cpu")


DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")
