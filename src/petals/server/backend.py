"""Code for serving bloom blocks via hivemind-server"""
from typing import Any, Dict, Sequence, Tuple

import torch
from hivemind import BatchTensorDescriptor, use_hivemind_log_handler
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger

from petals.bloom.from_pretrained import BloomBlock
from petals.server.memory_cache import MemoryCache
from petals.server.task_pool import PrioritizedTaskPool
from petals.utils.misc import is_dummy

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class TransformerBackend(ModuleBackend):
    """A wrapper for BloomBlock that can process requests for bloom layer forward, forward_incremental, and backward"""

    def __init__(self, *args, memory_cache: MemoryCache, backend_dtype: torch.dtype, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.module, BloomBlock)
        self.memory_cache = memory_cache
        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Bloom layer parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Bloom layer parameters must not accumulate gradients, but {name} does"

        max_batch_size = self.forward_pool.max_batch_size
        self.inference_pool = PrioritizedTaskPool(
            self.inference_step, max_batch_size=max_batch_size, name=f"{self.name}_inference"
        )
        self.forward_pool = PrioritizedTaskPool(
            self.forward, max_batch_size=max_batch_size, name=f"{self.name}_forward"
        )
        self.backward_pool = PrioritizedTaskPool(
            self.backward, max_batch_size=max_batch_size, name=f"{self.name}_backward"
        )

        assert backend_dtype is not None
        self.dtype = backend_dtype
        self.inference_schema = (
            (
                *self.args_schema,
                BatchTensorDescriptor((), dtype=self.dtype),
                BatchTensorDescriptor((), dtype=torch.int64),
            ),
            self.kwargs_schema,
        )

    def inference_step(self, cache_metadata: torch.IntTensor, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with torch.inference_mode():
            attention_cache_handle = int(cache_metadata[0, 0].item())
            prefix_length = int(cache_metadata[0, 1].item())
            (hidden_states, hypo_ids) = inputs
            assert (
                hidden_states.ndim == 3
            ), "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"

            with self.memory_cache.use_cache(attention_cache_handle) as cache:
                assert isinstance(self.module, BloomBlock) and cache.shape[0] == 2 and cache.ndim == 5
                if not is_dummy(hypo_ids):
                    assert hypo_ids.shape[0] == cache.shape[1]
                    cache[:, :] = cache[:, hypo_ids]  # in-place reorder cache by hypo ids
                layer_past = past_k, past_v = cache[0, :, :prefix_length], cache[1, :, :prefix_length]
                logger.debug(f"Metadata: {cache_metadata}, past_k.shape={past_k.shape}, past_v.shape={past_v.shape}")
                hidden_states, (new_k, new_v) = self.module.forward(
                    hidden_states, layer_past=layer_past, use_cache=True
                )

                # todo remove these asserts once we pass all tests
                new_length = new_v.shape[1]
                assert new_length > prefix_length
                assert new_k.shape[0] == past_k.shape[0] and new_v.shape[0] == past_v.shape[0]
                assert new_k.shape[1] == new_length and new_v.shape[1] == new_length
                assert new_k.shape[2:] == past_k.shape[2:] and new_v.shape[2:] == past_v.shape[2:]
                cache[0, :, prefix_length:new_length, :] = new_k[:, prefix_length:new_length]
                cache[1, :, prefix_length:new_length, :] = new_v[:, prefix_length:new_length]
                return (hidden_states,)

    def get_pools(self) -> Sequence[PrioritizedTaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool

    def get_info(self) -> Dict[str, Any]:
        """Get module parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(super().get_info(), inference_schema=self.inference_schema)
