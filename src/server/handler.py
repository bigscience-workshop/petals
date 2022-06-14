from typing import AsyncIterator

from hivemind import P2PContext
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.proto import runtime_pb2


class BloomConnectionHandler(ConnectionHandler):
    """Handles three request types: forward, backward and forward-incremental (inference)"""

    async def rpc_forward_incremental(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:
        # encode expert_uid as @model_name[starting_layer:finishing_layer]
        # - while not closed: read input embeddings, check input shapes, run inference, return batch of outputs, repeat
        # - receive and maintain a handle for attention cache here

        raise NotImplementedError()
