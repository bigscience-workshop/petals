from typing import AsyncIterator, Dict

from hivemind import P2PContext, DHT
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.proto import runtime_pb2

from src.bloom.block import BloomBlock


class TransformerConnectionHandler(ConnectionHandler):
    """Handles three request types: forward, backward and forward-incremental (inference)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def rpc_forward_incremental(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:
        # note: you may use self.experts[uid].memory_cache!
        # encode expert_uid as @model_name[starting_layer:finishing_layer]
        # - while not closed: read input embeddings, check input shapes, run inference, return batch of outputs, repeat
        # - receive and maintain a handle for attention cache here

        raise NotImplementedError()
