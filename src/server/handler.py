# Note: this code is being actively modified by justheuristic. If you want to change anything about it, please warn me.
import contextlib
from typing import AsyncIterator, Dict, Sequence

import torch
from hivemind import DHT, P2PContext, TensorDescriptor, deserialize_torch_tensor, nested_flatten, serialize_torch_tensor
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import anext

from src.data_structures import CHAIN_DELIMITER, ModuleUID
from src.server.backend import MAX_LENGTH, TransformerBackend


class TransformerConnectionHandler(ConnectionHandler):
    """Handles three request types: forward, backward and forward-incremental (inference)"""
    module_backends: Dict[ModuleUID, TransformerBackend]

    def __init__(self, dht: DHT, module_backends: Dict[str, TransformerBackend]):
        super().__init__(dht, module_backends)
        for module_backend in self.module_backends.values():
            assert isinstance(module_backend, TransformerBackend)

    async def rpc_inference(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:
        """Compute a single step of inference using attention cache; update attention cache accordingly."""
        try:
            print("OPENED RPC_INFERENCE")
            request = await anext(requests)
            requested_uids = self._check_header(request)
            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)

            cache_metadata = torch.tensor([[-1, -1]], dtype=torch.int64)  # [cache_handle, prefix_length]
            prefix_length = 0

            async with self._allocate_caches(requested_backends) as cache_handles:
                assert len(cache_handles) == len(requested_backends)
                while request.tensors:  # iterate while user is willing to supply tensors
                    hidden_states = [deserialize_torch_tensor(tensor) for tensor in request.tensors]

                    # run request tensors through all requested modules, update caches
                    for backend, cache_handle in zip(requested_backends, cache_handles):
                        cache_metadata[0, 0], cache_metadata[0, 1] = cache_handle, prefix_length
                        assert len(hidden_states) == 1 and hidden_states[0].ndim == 3, \
                            f"inputs to {type(backend)} must be a list with a single 3d tensor of hidden states"

                        hidden_states = await backend.inference_pool.submit_task(cache_metadata, *hidden_states)
                        assert isinstance(hidden_states, (list, tuple))
                        assert len(hidden_states) == 1 and hidden_states[0].ndim == 3

                    # serialize and send last layer outputs
                    yield runtime_pb2.ExpertResponse(tensors=[
                        serialize_torch_tensor(result, proto.compression, allow_inplace=True)
                        for result, proto in zip(hidden_states, nested_flatten(requested_backends[-1].outputs_schema))
                    ])

                    # prepare for next step
                    prefix_length += hidden_states[0].shape[1]
                    request = await (anext(requests))
        finally:
            print("CLOSED RPC_INFERENCE")

    def _check_header(self, request: runtime_pb2.ExpertRequest) -> Sequence[ModuleUID]:
        """Check that the first request to rpc_inference is valid"""
        uids = (request.uid or '').split(CHAIN_DELIMITER)
        if not uids:
            raise RuntimeError("User did not provide any uids")
        for uid in uids:
            if uid not in self.module_backends:
                raise RuntimeError(f"Remote peer does not serve {uid}")
        return tuple(uids)

    @contextlib.asynccontextmanager
    async def _allocate_caches(self, backends: Sequence[TransformerBackend]) -> Sequence[int]:
        """Allocate memory caches for each transformer block, return cache handles"""
        async with contextlib.AsyncExitStack() as stack:
            handles = []
            for backend in backends:
                num_heads = backend.module.self_attention.num_heads
                head_dim = backend.module.self_attention.head_dim

                cache_descriptor = TensorDescriptor(size=(2, 1, MAX_LENGTH, num_heads, head_dim), dtype=torch.float32)
                # [key_or_value, batch_size, max_length, num_heads, head_dim]

                handles.append(await stack.enter_async_context(backend.memory_cache.allocate_cache(cache_descriptor)))

            yield handles








