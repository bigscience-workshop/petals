from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvTransposeNd

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
