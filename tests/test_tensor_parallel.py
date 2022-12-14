import random

import pytest
import torch
from tensor_parallel import Config, TensorParallel
from test_utils import MODEL_NAME

from petals.bloom.from_pretrained import load_pretrained_block


@pytest.mark.parametrize("custom_config", [True, False])
@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3, ("cpu",) * 4])
def test_tp_block(devices, custom_config):
    block_index = random.randint(0, 10)
    block = load_pretrained_block(MODEL_NAME, block_index=block_index, torch_dtype=torch.float32).to(devices[0])

    tp_config = None
    if custom_config:
        tp_config = Config(
            state_rules={
                r".*self_attention\.query_key_value\.(weight|bias)": "split 0",
                r".*self_attention\.dense\.(weight|bias)": "split 0",
                r".*mlp\.dense_h_to_4h\.(weight|bias)": "split 0",
                r".*mlp\.dense_4h_to_h\.weight": "split 1",
                r".*mlp\.dense_4h_to_h\.bias": "scale",
            },
            input_rules={},
            output_rules={
                ".*self_attention\.query_key_value": {0: "gather -1"},
                ".*self_attention\.dense": {0: "gather -1"},
                ".*mlp\.dense_4h_to_h$": {0: "sum"},
            },
            attr_rules={},
        )

    test_inputs1 = torch.randn(2, 3, 1024, requires_grad=True, device=devices[0])
    test_inputs2 = test_inputs1.detach().clone().requires_grad_(True)
    batch_size = test_inputs1.shape[0]
    head_dim = len(block.input_layernorm.weight) // block.num_heads
    prefix_length = 5

    layer_past = (
        torch.randn(batch_size * block.num_heads, head_dim, prefix_length, device=devices[0]),
        torch.randn(batch_size * block.num_heads, prefix_length, head_dim, device=devices[0]),
    )

    grad_proj = torch.rand_like(test_inputs1)
    y_ref, cache_ref = block(test_inputs1, use_cache=True, layer_past=layer_past)
    y_ref.backward(grad_proj)

    block_tp = TensorParallel(block, devices, config=tp_config)
    y_ours, cache_ours = block_tp(test_inputs2, use_cache=True, layer_past=layer_past)
    y_ours.backward(grad_proj)

    assert torch.allclose(y_ours, y_ref, atol=1e-6)
    assert torch.allclose(test_inputs1.grad, test_inputs2.grad, atol=1e-5)
    assert torch.allclose(cache_ref[0], cache_ours[0], atol=1e-6)
    assert torch.allclose(cache_ref[1], cache_ours[1], atol=1e-6)
