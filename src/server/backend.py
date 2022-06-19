"""Code for serving bloom blocks via hivemind-server"""
from typing import Sequence, Tuple

import torch
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.moe.server.task_pool import TaskPool

from src.bloom.block import BloomBlock
from src.server.cache import MemoryCache

MAX_LENGTH = 2048


class TransformerBackend(ModuleBackend):
    """A wrapper for BloomBlock that can process requests for bloom layer forward, forward_incremental, and backward"""

    def __init__(self, *args, memory_cache: MemoryCache, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.module, BloomBlock)
        self.memory_cache = memory_cache
        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Bloom layer parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Bloom layer parameters must not accumulate gradients, but {name} does"

        self.inference_pool = TaskPool(self.inference_step, max_batch_size=1, name=f"{self.name}_inference")

    def inference_step(self, cache_metadata: torch.IntTensor, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        attention_cache_handle = int(cache_metadata[0, 0].item())
        prefix_length = int(cache_metadata[0, 1].item())
        (
            hidden_states,
            *_,
        ) = inputs  # todo: this ignores any extra inputs for now; in future, it would be best to support attention mask as an extra input
        assert hidden_states.ndim == 3, "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"

        with self.memory_cache.use_cache(attention_cache_handle) as cache:
            print("METADATA:", cache_metadata)
            assert isinstance(self.module, BloomBlock) and cache.shape[0] == 2 and cache.ndim == 5
            layer_past = past_k, past_v = cache[0, :, :prefix_length], cache[1, :, :prefix_length]
            print(past_k.shape, past_v.shape)
            hidden_states, (new_k, new_v) = self.module.forward(hidden_states, layer_past=layer_past, use_cache=True)

            # todo remove these debugprints
            new_length = new_v.shape[1]
            assert new_length > prefix_length
            assert new_k.shape[0] == past_k.shape[0] and new_v.shape[0] == past_v.shape[0]
            assert new_k.shape[1] == new_length and new_v.shape[1] == new_length
            assert new_k.shape[2:] == past_k.shape[2:] and new_v.shape[2:] == past_v.shape[2:]
            assert torch.allclose(new_v[:, : past_v.shape[1]], past_v)
            assert torch.allclose(new_k[:, : past_k.shape[1]], past_k)
            cache[0, :, prefix_length:new_length, :] = new_k[:, prefix_length:new_length]
            cache[1, :, prefix_length:new_length, :] = new_v[:, prefix_length:new_length]
            return (hidden_states,)

    def get_pools(self) -> Sequence[TaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool
