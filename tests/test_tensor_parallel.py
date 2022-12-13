import random
from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from test_utils import MODEL_NAME
from torch.nn.modules.conv import _ConvTransposeNd

from petals.bloom.from_pretrained import load_pretrained_block
from petals.utils.tensor_parallel import TensorParallel


@pytest.mark.parametrize("devices", [None, ("cpu",), ("cpu", "cpu"), ("cpu", "cpu", "cpu")])
def test_embeds_and_linear(devices):
    for emb_cls in nn.Embedding, nn.EmbeddingBag:
        model = nn.Sequential(
            emb_cls(num_embeddings=1337, embedding_dim=64),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        inputs = torch.randint(1, 1000, size=(1, 10))
        ref_out = model(inputs)
        ref_out.norm().backward()

        model_tp = deepcopy(model)  # deepcopy to avoid accidental grad spillage and false positives
        model_tp = TensorParallel(model_tp, device_ids=devices)
        out_ours = model_tp(inputs)
        out_ours.norm().backward()
        assert torch.allclose(ref_out, out_ours, atol=1e-6)
        our_grad = torch.cat([next(shard[0].parameters()).grad for shard in model_tp.module_shards], dim=1)
        assert torch.allclose(model[0].weight.grad, our_grad, atol=1e-6)


@pytest.mark.parametrize("devices", [None, ("cpu",), ("cpu",) * 2, ("cpu",) * 3, ("cpu",) * 4])
@pytest.mark.parametrize("extra_options", [{}, {"padding": "same"}, {"stride": 2}, {"dilation": 2}])
def test_convs(devices, extra_options):
    for Conv, nd in (
        (nn.Conv1d, 1),
        (nn.Conv2d, 2),
        (nn.Conv3d, 3),
        (nn.ConvTranspose1d, 1),
        (nn.ConvTranspose2d, 2),
        (nn.ConvTranspose3d, 3),
    ):
        if issubclass(Conv, _ConvTransposeNd) and "padding" in extra_options:
            continue  # unsupported by pytorch
        model = nn.Sequential(
            Conv(32, 64, kernel_size=(3,) * nd, **extra_options),
            nn.ReLU(),
            Conv(64, 14, kernel_size=(3,) * nd, **extra_options),
        )
        inputs = torch.randn(3, 32, *[10 for _ in range(nd)])
        ref_out = model(inputs)
        ref_out.norm().backward()

        model_tp = deepcopy(model)  # deepcopy to avoid accidental grad spillage and false positives
        model_tp = TensorParallel(model_tp, device_ids=devices)
        out_ours = model_tp(inputs)
        out_ours.norm().backward()
        assert torch.allclose(ref_out, out_ours, atol=1e-6)
        dim = 1 if model[0].transposed else 0  # concat over output channels
        our_grad = torch.cat([next(shard[0].parameters()).grad for shard in model_tp.module_shards], dim=dim)
        assert torch.allclose(model[0].weight.grad, our_grad, atol=1e-6)


@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3, ("cpu",) * 4])
def petals_test_tp_block(devices):
    block_index = random.randint(0, 10)
    block = load_pretrained_block(MODEL_NAME, block_index=block_index, torch_dtype=torch.float32).to(devices[0])

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

    block_tp = TensorParallel(block, devices)
    y_ours, cache_ours = block_tp(test_inputs2, use_cache=True, layer_past=layer_past)
    y_ours.backward(grad_proj)

    assert torch.allclose(y_ours, y_ref, atol=1e-6)
    assert torch.allclose(test_inputs1.grad, test_inputs2.grad, atol=1e-6)
    assert torch.allclose(cache_ref[0], cache_ours[0], atol=1e-6)
    assert torch.allclose(cache_ref[1], cache_ours[1], atol=1e-6)
