"""Code for serving bloom blocks via hivemind-server"""
from typing import Tuple

import torch
from hivemind import BatchTensorDescriptor
from hivemind.moe.server.expert_backend import ExpertBackend
from hivemind.moe.server.task_pool import TaskPool

from src.bloom.block import BloomBlock
from src.server.cache import MemoryCache


# TODO
# BloomBackend serves a single layer
# - ensure that parameters do not require grad!
# - ensure that TaskPool for inference is NOT batched
# - ensure that optimizer/scheduler is not created


class BloomBlockBackend(ExpertBackend):
    """A wrapper for BloomBlock that can process requests for bloom layer forward, forward_incremental, and backward"""
    def __init__(self, name: str, module: BloomBlock, *, memory_cache: MemoryCache, **kwargs):
        object().__init__()  # to bypass super.__init__
        self.name, self.module = name, module
        self.memory_cache = memory_cache

        for name, param in module.named_parameters():
            assert not param.requires_grad, f"Bloom layer parameters must not accumulate gradients, but {name} does"
        for name, buf in module.named_buffers():
            assert not buf.requires_grad, f"Bloom layer parameters must not accumulate gradients, but {name} does"

        self.args_schema = (BatchTensorDescriptor(HARDCODCED_LENGTH, module.hidden_size),)
        self.kwargs_schema = {}
        self.outputs_schema = (BatchTensorDescriptor(HARDCODCED_LENGTH, module.hidden_size),)

        self.forward_schema = (self.args_schema, self.kwargs_schema)  # inputs for forward
        self.backward_schema = (self.forward_schema, self.outputs_schema)  # inputs to backward

        self.grad_inputs_schema = self.forward_schema  # outputs from backward have same shape as inputs for forward
        self.forward_pool = TaskPool(self.forward, name=f"{self.name}_forward", **kwargs)
        self.backward_pool = TaskPool(self.backward, name=f"{self.name}_backward", **kwargs)

    @property
    def expert(self):
        #TODO un-hardcode this naming from hivemind
        return self.module

    def forward_incremental(self, *inputs: torch.Tensor, attention_cache_handle: int) -> Tuple[torch.Tensor, ...]:
        with self.memory_cache.use_cache(attention_cache_handle) as (current_length, cached_keys, cached_values):
            raise NotImplementedError("TODO")

