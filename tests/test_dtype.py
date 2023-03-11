import pytest
import torch

from petals.bloom.from_pretrained import load_pretrained_block
from petals.client import DistributedBloomConfig
from petals.server.block_utils import resolve_block_dtype

MODEL_NAME = "bloom-testing/test-bloomd-350m-main"


@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float16, "auto"])
def test_backend_dtype(torch_dtype):
    config = DistributedBloomConfig.from_pretrained(MODEL_NAME)
    block = load_pretrained_block(MODEL_NAME, 0, config, torch_dtype=torch_dtype)
    backend_dtype = resolve_block_dtype(config, torch_dtype)
    other_backend_dtype = next(block.parameters()).dtype if torch_dtype == "auto" else torch_dtype
    assert backend_dtype == other_backend_dtype
