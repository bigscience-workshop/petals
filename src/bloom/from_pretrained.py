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
from hivemind.utils.logging import use_hivemind_log_handler, get_logger
from transformers.utils.hub import hf_bucket_url, cached_path

from src.bloom import BloomForCausalLM, DistributedBloomConfig, BloomBlock
from transformers.modeling_utils import WEIGHTS_NAME

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)

CLIENT_BRANCH = "client"
BLOCK_BRANCH_PREFIX = "block_"
USER_AGENT = {'file_type': 'model', 'framework': 'pytorch', 'from_auto_class': False}
cls = BloomForCausalLM
FORCE_DOWNLOAD = False
RESUME_DOWNLOAD = False
LOCAL_FILES_ONLY = False


def load_pretrained_block(
        converted_model_name_or_path: str, block_index: int,
        config: Optional[DistributedBloomConfig] = None, torch_dtype: Union[torch.dtype, str] = 'auto') -> BloomBlock:
    """Load one BloomBlock from a converted model. See convert_model.py (or README.md) on how to convert it."""
    if config is None:
        config = DistributedBloomConfig.from_pretrained(converted_model_name_or_path)
    block = BloomBlock(config, layer_number=block_index)
    state_dict = _load_state_dict(converted_model_name_or_path, block_index)
    block.load_state_dict(state_dict)

    if torch_dtype == 'auto':
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
        pretrained_model_name_or_path: str, block_index: Optional[int] = None) -> OrderedDict[str, torch.Tensor]:
    revision = BLOCK_BRANCH_PREFIX + str(block_index) if block_index is not None else CLIENT_BRANCH
    archive_file = hf_bucket_url(pretrained_model_name_or_path, filename=WEIGHTS_NAME, revision=revision, mirror=None)

    # Load from URL or cache if already cached
    resolved_archive_file = cached_path(
        archive_file,
        cache_dir=None,
        force_download=FORCE_DOWNLOAD,
        proxies=None,
        resume_download=RESUME_DOWNLOAD,
        local_files_only=LOCAL_FILES_ONLY,
        use_auth_token=True,
        user_agent=USER_AGENT,
    )
    state_dict = torch.load(resolved_archive_file, map_location='cpu')
    return state_dict


DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")
