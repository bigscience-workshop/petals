from concurrent.futures import Future
from functools import partial
from typing import List, Optional, Union, Sequence

import torch
from hivemind.moe.client import RemoteExpert
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.moe.expert_uid import ExpertUID
from hivemind.moe.server.dht_handler import _get_experts
from hivemind.p2p import StubBase, P2P
from hivemind.proto.runtime_pb2 import ExpertInfo
from hivemind.dht import DHT
from hivemind.utils import MPFuture, DHTExpiration

from src.server.handler import TransformerConnectionHandler


class RemoteTransformerBlock(RemoteExpert):
    """A class that interacts with a specific remote server for forward/backward or inference"""

    def __init__(self, info: ExpertInfo, p2p: P2P):
        super().__init__(info, p2p)
        # self._config = config
        # self._inputs_cache = torch.empty(1, MAX_LENGTH, config.hidden_size, dtype=config.dtype)
        # self._active_stream: Optional[RemoteTransformerStream] = None

    @property
    def stub(self) -> StubBase:
        return TransformerConnectionHandler.get_stub(self.p2p, self.peer_id)



def get_remote_module(
    dht: DHT, uids: List[ExpertUID], expiration_time: Optional[DHTExpiration] = None, return_future: bool = False
) -> Union[List[Optional[RemoteTransformerBlock]], MPFuture[List[Optional[RemoteTransformerBlock]]]]:
    """
    :param uids: find experts with these ids from across the DHT
    :param expiration_time: if specified, return experts that expire no sooner than this (based on get_dht_time)
    :param return_future: if False (default), return when finished. Otherwise return MPFuture and run in background.
    :returns: a list of [RemoteTransformerBlock if found else None]
    """
    assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
    result = dht.run_coroutine(partial(_get_experts, uids=list(uids), expiration_time=expiration_time), return_future)
    return create_remote_module(result, dht, return_future)


def create_remote_module(
    infos: Union[Sequence[Optional[ExpertInfo]], MPFuture], dht: DHT, return_future: bool = False
) -> Union[List[Optional[RemoteTransformerBlock]], Future]:
    if return_future:

        async def _unpack(infos_future: MPFuture, dht: DHT):
            p2p = await dht.replicate_p2p()
            return _create_remote_experts(await infos_future, p2p)

        return RemoteExpertWorker.run_coroutine(_unpack(infos, dht), return_future)
    p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p())
    return _create_remote_experts(infos, p2p)


def _create_remote_experts(infos: Sequence[Optional[ExpertInfo]], p2p: P2P) -> List[Optional[RemoteTransformerBlock]]:
    experts: List[Optional[RemoteTransformerBlock]] = []
    for info in infos:
        if info is not None:
            experts.append(RemoteTransformerBlock(info, p2p))
        else:
            experts.append(None)
    return experts
