from typing import AsyncIterator, Dict

import torch
from hivemind import P2PContext, DHT, deserialize_torch_tensor, TensorDescriptor, nested_flatten
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import anext

from src.server.backend import TransformerBackend, MAX_LENGTH


class TransformerConnectionHandler(ConnectionHandler):
    """Handles three request types: forward, backward and forward-incremental (inference)"""

    def __init__(self, dht: DHT, module_backends: Dict[str, TransformerBackend]):
        for module_backend in module_backends.values():
            assert isinstance(module_backend, TransformerBackend)
        super().__init__(dht, module_backends)

    async def rpc_inference(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:

        request = await anext(requests)
        backend = self.module_backends[request.uid]
        assert isinstance(backend, TransformerBackend)

        inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]

        hidden_size = backend.module.hidden_size
        cache_descriptor = TensorDescriptor(size=(1, MAX_LENGTH, hidden_size), dtype=torch.float32)
        async with backend.memory_cache.allocate_cache(cache_descriptor) as handle:
            inputs.insert(0, torch.tensor([handle], dtype=torch.int64))
            outputs = await self._process_inputs(inputs, backend.inference_pool, backend.outputs_schema)

        yield runtime_pb2.ExpertResponse(tensors=outputs)
