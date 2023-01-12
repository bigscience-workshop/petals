"""Code for serving bloom blocks via hivemind-server"""
from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Sequence, Tuple

import torch
from hivemind import BatchTensorDescriptor, TensorDescriptor
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from tensor_parallel import TensorParallel
from tensor_parallel.tensor_parallel import PerDeviceTensors
from transformers import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomAttention

from petals.data_structures import InferenceMetadata
from petals.server.memory_cache import MemoryCache
from petals.server.task_pool import PrioritizedTaskPool
from petals.utils.misc import is_dummy

logger = get_logger(__file__)


class TransformerBackend(ModuleBackend):
    """A wrapper for a BLOOM block that can process requests for BLOOM layer forward, backward and inference"""

    def __init__(self, *args, config: BloomConfig, memory_cache: MemoryCache, backend_dtype: torch.dtype, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.module, TensorParallel)
        self.config = config
        self.memory_cache = memory_cache
        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Bloom layer parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Bloom layer parameters must not accumulate gradients, but {name} does"

        max_batch_size = self.forward_pool.max_batch_size
        device = self.module.devices[self.module.output_device_index]
        self.inference_pool = PrioritizedTaskPool(
            self.inference_step, max_batch_size=max_batch_size, device=device, name=f"{self.name}_inference"
        )
        self.forward_pool = PrioritizedTaskPool(
            self.forward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_forward"
        )
        self.backward_pool = PrioritizedTaskPool(
            self.backward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_backward"
        )

        assert backend_dtype is not None
        self.dtype = backend_dtype
        self.shard_num_heads = []
        for shard in self.module.module_shards:
            for submodule in shard.modules():
                if isinstance(submodule, BloomAttention):
                    self.shard_num_heads.append(submodule.num_heads)
        assert len(self.shard_num_heads) == len(self.module.devices) and sum(self.shard_num_heads) == config.n_head

        self.inference_schema = (
            (
                *self.args_schema,
                BatchTensorDescriptor((), dtype=self.dtype),
                BatchTensorDescriptor((), dtype=torch.int64),
            ),
            self.kwargs_schema,
        )

        self.cache_bytes_per_token: Dict[torch.device, int] = Counter()
        for descr in self.get_inference_cache_descriptors(batch_size=1, max_length=1):
            self.cache_bytes_per_token[descr.device] += descr.numel() * torch.finfo(descr.dtype).bits // 8

    def get_inference_cache_descriptors(self, batch_size: int, max_length: int) -> Sequence[TensorDescriptor]:
        """Create tensor descriptors for attention cache tensors used during inference_step"""
        head_dim = self.config.hidden_size // self.config.n_head
        cache_tensors = []
        for device, num_heads in zip(self.module.devices, self.shard_num_heads):
            keys = TensorDescriptor((batch_size, num_heads, head_dim, max_length), dtype=self.dtype, device=device)
            values = TensorDescriptor((batch_size, num_heads, max_length, head_dim), dtype=self.dtype, device=device)
            cache_tensors.extend((keys, values))
        return cache_tensors

    def inference_step(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        inference_info: InferenceMetadata,
    ) -> Tuple[torch.Tensor, ...]:
        with torch.inference_mode():
            assert (
                hidden_states.ndim == 3
            ), "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"
            with self.memory_cache.use_cache(*inference_info.cache_handles) as cache_tensors:
                self._reorder_cache_inplace(cache_tensors, hypo_ids)
                layer_past = self._select_layer_past(cache_tensors, inference_info.prefix_length)
                hidden_states, new_kvs = self.module.forward(hidden_states, layer_past=layer_past, use_cache=True)
                self._update_cache_inplace(cache_tensors, new_kvs, inference_info.prefix_length)
                return (hidden_states,)

    def _reorder_cache_inplace(self, cache_tensors: torch.Tensor, hypo_ids: torch.Tensor):
        """If hypo_ids is specified, reorder elements of each cache tensor in-place by taking indices from hypo_ids"""
        if not is_dummy(hypo_ids):
            for cache_tensor in cache_tensors:
                cache_tensor[...] = cache_tensor[hypo_ids]  # in-place reorder cache by hypo ids

    def _select_layer_past(self, cache_tensors: Sequence[torch.Tensor], prefix_length: int) -> Sequence[torch.Tensor]:
        """Extract first {prefix_length} tokens and reshape them such that they can be used as layer_past"""
        key_cache, value_cache = list(cache_tensors[0::2]), list(cache_tensors[1::2])
        for i in range(len(key_cache)):
            key_cache[i] = key_cache[i].flatten(0, 1)[:, :, :prefix_length]  # [batch * num_heads, head_dim, kv_length]
            value_cache[i] = value_cache[i].flatten(0, 1)[:, :prefix_length]  # [batch * num_heads, kv_length, head_dim]
        layer_past = tuple(chain(*zip(key_cache, value_cache)))
        return PerDeviceTensors(*layer_past) if len(self.module.module_shards) > 1 else layer_past

    def _update_cache_inplace(
        self, cache_tensors: Sequence[torch.Tensor], new_kvs: Sequence[torch.Tensor], prefix_length: int
    ):
        """Writes new key/value tensors back into cache, works in-place"""
        _batch_size_times_num_heads, head_dim, new_length = new_kvs[0].shape
        for cache_key, new_key in zip(cache_tensors[0::2], new_kvs[0::2]):
            new_key = new_key.view(*cache_key.shape[:3], new_length)
            cache_key[:, :, :, prefix_length:new_length] = new_key[:, :, :, prefix_length:new_length]
        for cache_value, new_value in zip(cache_tensors[1::2], new_kvs[1::2]):
            new_value = new_value.view(*cache_value.shape[:2], new_length, head_dim)
            cache_value[:, :, prefix_length:new_length, :] = new_value[:, :, prefix_length:new_length, :]

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
