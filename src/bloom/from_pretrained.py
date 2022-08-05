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
from transformers.utils.hub import cached_path, hf_bucket_url

from src.bloom import BloomBlock, BloomConfig

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
) -> BloomBlock:
    """Load one BloomBlock from a converted model. See convert_model.py (or README.md) on how to convert it."""
    if config is None:
        config = BloomConfig.from_pretrained(converted_model_name_or_path, use_auth_token=use_auth_token)
    block = BloomBlock(config, layer_number=block_index)
    state_dict = _load_state_dict(
        converted_model_name_or_path, block_index, use_auth_token=use_auth_token, cache_dir=cache_dir
    )
    block.load_state_dict(state_dict)

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
    block_index: Optional[int] = None,
    use_auth_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> OrderedDict[str, torch.Tensor]:
    revision = BLOCK_BRANCH_PREFIX + str(block_index) if block_index is not None else CLIENT_BRANCH
    archive_file = hf_bucket_url(pretrained_model_name_or_path, filename=WEIGHTS_NAME, revision=revision, mirror=None)

    # Load from URL or cache if already cached
    resolved_archive_file = cached_path(
        archive_file,
        cache_dir=cache_dir,
        force_download=FORCE_DOWNLOAD,
        proxies=None,
        resume_download=RESUME_DOWNLOAD,
        local_files_only=LOCAL_FILES_ONLY,
        use_auth_token=use_auth_token,
        user_agent=USER_AGENT,
    )
    state_dict = torch.load(resolved_archive_file, map_location="cpu")
    return state_dict


DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")
