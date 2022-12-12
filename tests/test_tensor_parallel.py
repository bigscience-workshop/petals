from copy import deepcopy

import pytest
import torch
import torch.nn as nn

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

        model_tp = deepcopy(model)  # deepcopy to avoid accidental grad spilage and false positives
        model_tp = TensorParallel(model_tp, device_ids=devices)
        out_ours = model_tp(inputs)
        out_ours.norm().backward()
        assert torch.allclose(ref_out, out_ours, atol=1e-6)
        our_grad = torch.cat([next(shard[0].parameters()).grad for shard in model_tp.module_shards], dim=1)
        assert torch.allclose(model[0].weight.grad, our_grad, atol=1e-6)
