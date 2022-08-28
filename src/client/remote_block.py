# Note: this code is being actively modified by justheuristic. If you want to change anything about it, please warn me.
from __future__ import annotations

import random

import torch
from hivemind.moe.client.expert import RemoteExpert, RemoteExpertWorker
from hivemind.moe.expert_uid import ExpertInfo
from hivemind.p2p import P2P, StubBase
from hivemind.utils import get_logger, use_hivemind_log_handler

from src.client.inference_session import RemoteTransformerBlockInferenceSession
from src.data_structures import RemoteModuleInfo
from src.server.handler import TransformerConnectionHandler

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class RemoteTransformerBlock(RemoteExpert):
    """A class that interacts with a remote module on a specific server for forward/backward or inference"""

    def __init__(self, peers_info: RemoteModuleInfo, p2p: P2P):
        peer_info = ExpertInfo(peers_info.uid, random.choice(list(peers_info.servers.keys())))  # TODO replace this
        super().__init__(peer_info, p2p)

    @property
    def stub(self) -> StubBase:
        return TransformerConnectionHandler.get_stub(self.p2p, self.peer_id)

    def forward(self, inputs: torch.Tensor, **kwargs):
        for k, v in kwargs.items():
            assert v is None or v is False, f"Extra keyword arguments are not yet supported (got {k} = {v})"
        return super().forward(inputs)

    def inference_session(self, **kwargs) -> RemoteTransformerBlockInferenceSession:
        """Initialize a new inference session with the specified remote server"""
        return RemoteExpertWorker.run_coroutine(
            RemoteTransformerBlockInferenceSession._create(self.stub, self.uid, self.info, **kwargs)
        )
