"""Code for serving bloom blocks via hivemind-server"""
from typing import Any, Dict, Sequence, Tuple

import torch
from hivemind import BatchTensorDescriptor
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger

from petals.bloom.block import WrappedBloomBlock
from petals.server.memory_cache import MemoryCache
from petals.server.task_pool import PrioritizedTaskPool
from petals.utils.misc import is_dummy

logger = get_logger(__file__)


class TransformerBackend(ModuleBackend):
    """A wrapper for a BLOOM block that can process requests for BLOOM layer forward, backward and inference"""

    def __init__(self, *args, memory_cache: MemoryCache, backend_dtype: torch.dtype, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.module, WrappedBloomBlock)
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

    def inference_step(
        self, hidden_states: torch.Tensor, hypo_ids: torch.LongTensor, cache_metadata: torch.LongTensor
    ) -> Tuple[torch.Tensor, ...]:
        num_heads, head_dim = self.module.self_attention.num_heads, self.module.self_attention.head_dim
        with torch.inference_mode():
            assert (
                hidden_states.ndim == 3
            ), "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"
            cache_handle, rel_index, prefix_length = map(int, cache_metadata[0])

            with self.memory_cache.use_cache(cache_handle) as cache:
                batch_size = cache.shape[2]
                max_length = cache.shape[-1] // (head_dim * num_heads)
                assert isinstance(self.module, WrappedBloomBlock) and cache.shape[1] == 2 and cache.ndim == 4
                if not is_dummy(hypo_ids):
                    assert hypo_ids.shape[0] == batch_size
                    cache[rel_index, :, :] = cache[rel_index, :, hypo_ids]  # in-place reorder cache by hypo ids
                key_cache = cache[rel_index, 0].view(batch_size, num_heads, head_dim, max_length)
                value_cache = cache[rel_index, 1].view(batch_size, num_heads, max_length, head_dim)

                key_past = key_cache.flatten(0, 1)[:, :, :prefix_length]  # [batch * num_heads, head_dim, kv_length]
                value_past = value_cache.flatten(0, 1)[:, :prefix_length, :]  # [batch * num_heads, kv_length, head_dim]
                logger.debug(
                    f"Metadata: {cache_metadata}, past_k.shape={key_past.shape}, past_v.shape={value_past.shape}"
                )
                hidden_states, (new_key, new_value) = self.module.forward(
                    hidden_states, layer_past=(key_past, value_past), use_cache=True
                )
                new_length = new_key.shape[-1]
                assert new_length > prefix_length
                assert new_key.shape[0] == key_past.shape[0] and new_value.shape[0] == value_past.shape[0]
                assert new_key.shape[-1] == new_length and new_value.shape[-2] == new_length
                new_key = new_key.view(batch_size, num_heads, head_dim, -1)
                new_value = new_value.view(batch_size, num_heads, -1, head_dim)
                key_cache[:, :, :, prefix_length:new_length] = new_key[:, :, :, prefix_length:new_length]
                value_cache[:, :, prefix_length:new_length, :] = new_value[:, :, prefix_length:new_length, :]
                return (hidden_states,)

    def get_pools(self) -> Sequence[PrioritizedTaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool

    def get_info(self) -> Dict[str, Any]:
        """Get module parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(super().get_info(), inference_schema=self.inference_schema)

    def shutdown(self):
        # Break the cyclic references, otherwise TransformerBackend may be not garbage-collected
        self.forward_pool = self.backward_pool = self.inference_pool = None

        # Explicitly free the GPU memory. This is not necessary at the time this code is written,
        # but may help to avoid future issues when the module is not garbage-collected for some reasons
        dummy = torch.tensor([])
        for p in self.module.parameters():
            p.data = dummy
