"""Code for serving bloom blocks via hivemind-server"""
from typing import Tuple, Sequence

import torch
from hivemind.moe.server.expert_backend import ExpertBackend
from hivemind.moe.server.task_pool import TaskPool

from src.bloom.block import BloomBlock
from src.server.cache import MemoryCache


class TransformerBlockBackend(ExpertBackend):
    """A wrapper for BloomBlock that can process requests for bloom layer forward, forward_incremental, and backward"""

    def __init__(self, *args, memory_cache: MemoryCache, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_cache = memory_cache

        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Bloom layer parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Bloom layer parameters must not accumulate gradients, but {name} does"


        self.inference_pool = TaskPool(self.inference_step, max_batch_size=1, name=f"{self.name}_inference")

    def inference_step(self, *inputs: torch.Tensor, attention_cache_handle: int) -> Tuple[torch.Tensor, ...]:
        with self.memory_cache.use_cache(attention_cache_handle) as (current_length, cached_keys, cached_values):
            return inputs[0] * 2

    def get_pools(self) -> Sequence[TaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool
