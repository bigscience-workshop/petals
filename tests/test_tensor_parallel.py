import random

import pytest
import torch
import transformers
from tensor_parallel import TensorParallel
from tensor_parallel.slicing_configs import get_bloom_config

from petals.server.from_pretrained import load_pretrained_block
from test_utils import MODEL_NAME


@pytest.mark.forked
@pytest.mark.parametrize("custom_config", [True, False])
@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3, ("cpu",) * 4])
def test_tp_block(devices, custom_config):
    model_config = transformers.AutoConfig.from_pretrained(MODEL_NAME)
    if model_config.model_type != "bloom":
        pytest.skip("Tensor parallelism is implemented only for BLOOM for now")

    block_index = random.randint(0, 10)
    block = load_pretrained_block(MODEL_NAME, block_index=block_index, torch_dtype=torch.float32).to(devices[0])

    tp_config = None
    if custom_config:
        tp_config = get_bloom_config(model_config, devices)

    batch_size = 2
    prefix_length = 5

    test_inputs1 = torch.randn(batch_size, 3, 1024, requires_grad=True, device=devices[0])
    test_inputs2 = test_inputs1.detach().clone().requires_grad_(True)
    test_prefix1 = torch.randn(batch_size, prefix_length, 1024, requires_grad=True, device=devices[0])
    test_prefix2 = test_prefix1.detach().clone().requires_grad_(True)
    grad_proj = torch.rand_like(test_inputs1)

    y_prefix_ref, layer_past = block(test_prefix1, use_cache=True)
    y_ref, cache_ref = block(test_inputs1, use_cache=True, layer_past=layer_past)
    y_ref.backward(grad_proj)

    block_tp = TensorParallel(block, devices, config=tp_config)
    y_prefix, layer_past = block_tp(test_prefix2, use_cache=True)
    y_ours, cache_ours = block_tp(test_inputs2, use_cache=True, layer_past=layer_past)
    y_ours.backward(grad_proj)

    assert torch.allclose(y_prefix, y_prefix_ref, atol=1e-5)
    assert torch.allclose(y_ours, y_ref, atol=1e-5)
    assert torch.allclose(test_inputs1.grad, test_inputs2.grad, atol=1e-4)
    assert torch.allclose(test_prefix1.grad, test_prefix2.grad, atol=1e-4)
