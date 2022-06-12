"""Code for serving bloom blocks via hivemind-server"""
import contextlib
import threading
from typing import AsyncIterator, Tuple, List, Dict, Optional

import torch
from hivemind import P2PContext, DHT
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.moe.server.dht_handler import DHTHandlerThread
from hivemind.moe.server.expert_backend import ExpertBackend
from hivemind.moe.server.runtime import Runtime
from hivemind.moe.server.server import Server
from hivemind.proto import runtime_pb2
from torch import nn


class BloomServer(Server):
    """Serves one or more bloom layers for inference, forward and backward; announces oneself to the DHT"""
    def __init__(
            self, dht: DHT, device=torch.device, num_connection_handlers: int = 8, update_period: int = 30,
            attention_cache_size: Optional[int] = None, start=False, **kwargs,
    ):
        threading.Thread.__init__(self)
        self.attention_cache = AttentionCache(attention_cache_size, dtype=torch.bfloat16, device=torch.)
        expert_blocks = dict(LOAD_BLOOM_LAYERS_HERE)

        expert_backends = {name: _BloomBlockBackend(name, block, ..., self.attention_kv_cache) for name, block in expert_blocks.items()}
        self.dht, self.experts, self.update_period = dht, expert_backends, update_period
        self.conn_handlers = [ConnectionHandler(dht, self.experts) for _ in range(num_connection_handlers)]
        self.runtime = Runtime(self.experts, **kwargs)
        self.dht_handler_thread = DHTHandlerThread(self.experts, dht, update_period=update_period, daemon=True)
        self.checkpoint_saver = None  # no need to save checkpoints since we do not change model state

        if start:
            self.run_in_background(await_ready=True)


class _BloomConnectionHandler(ConnectionHandler):
    """Handles three request types: forward, backward and forward-incremental (inference)"""

    async def rpc_forward_incremental(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:
        # encode expert_uid as @model_name[starting_layer:finishing_layer]
        # - while not closed: read input embeddings, check input shapes, run inference, return batch of outputs, repeat
        # - receive and maintain a handle for attention cache here

        raise NotImplementedError()


class _BloomBlockBackend(ExpertBackend):
    def __init__(self, name: str, expert: nn.Module, *, attention_cache: AttentionCache, **kwargs):
        self.attention_cache = attention_cache
        super().__init__(name, expert, **kwargs)
        #TODO
        # BloomBackend serves a single layer
        # - ensure that parameters do not require grad!
        # - ensure that TaskPool for inference is NOT batched
        # - ensure that optimizer/scheduler is not created

    def forward_incremental(self, *inputs: torch.Tensor, attention_cache_handle: int) -> Tuple[torch.Tensor, ...]:
        with self.attention_cache.use_cache(attention_cache_handle) as (current_length, cached_keys, cached_values):
            raise NotImplementedError("TODO")


class AttentionCache:
    lock: mp.Lock
    data: Dict[int, SomeKindOfTupleWithTensors]  # workaround for now, while we are on CPU
    @contextlib.asynccontextmanager
    async def allocate_cache(self, size: torch.Size, dtype: torch.dtype) -> int:
        """
        Allocate buffers for attention cache on the compute device, return a unique handle;
        This function should be called by connection handler processes, may be called concurrently
        """
        try:
            async with acquire_asynchronously(self.lock):
                handle: int = generate_unique_handle() # or just use  counter mpvalue and increment each time
                assert handle not in data
                self.data[handle] = todo_allocate(self, size, dtype)
            yield handle
        finally:
            todo_deallocate(self, handle)
            # ^-- this should NOT move any data. But it may mark data for movement during next allocation
            self.data.pop(handle, None);

    def use_cache(self, handle: int) -> Tuple[mp.Value, torch.Tensor, torch.Tensor]:
        """Return a previously allocated cache, called by ExpertBackend in runtime (a single process)"""
        with self.lock:
            yield self.data[handle]



# later:
# - if possible, do not change how DHTHandler handles for now
# - do not worry about OOM in cache for now! - just make sure that nothing except cache could oom.
# - contiguous attention cache with max size
# - select a subset of experts
# - priorities
# - option to backtrack a few tokens
# - ensure that backprop is performed optimally, does not accumulate grads wrt parameters
# - forget about length-adaptive forward/backward for now, use fixed length, maybe several fixed lengths - or better yet, forget finetuning for now