from typing import AsyncIterator, Dict

import torch
from hivemind import P2PContext, DHT, deserialize_torch_tensor, TensorDescriptor
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import anext

from src.server.backend import TransformerBlockBackend


class TransformerConnectionHandler(ConnectionHandler):
    """Handles three request types: forward, backward and forward-incremental (inference)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def rpc_forward_incremental(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:

        request = await anext(requests)
        expert = self.experts[request.uid]
        assert isinstance(expert, TransformerBlockBackend)

        inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        async with expert.memory_cache.allocate_cache(TensorDescriptor.from_tensor(torch.randn(3))):
            outputs = await self._process_inputs(inputs, expert.forward_pool, expert.outputs_schema)

        return runtime_pb2.ExpertResponse(tensors=outputs)


        # note: you may use self.experts[uid].memory_cache!
        # encode expert_uid as @model_name[starting_layer:finishing_layer]
        # - while not closed: read input embeddings, check input shapes, run inference, return batch of outputs, repeat
        # - receive and maintain a handle for attention cache here

        raise NotImplementedError()
