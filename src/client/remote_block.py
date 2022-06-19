from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Union

import torch
from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.dht import DHT, DHTNode, DHTValue
from hivemind.moe.client.expert import RemoteExpert, RemoteExpertWorker
from hivemind.moe.expert_uid import ExpertInfo as RemoteModuleInfo
from hivemind.moe.expert_uid import ExpertUID
from hivemind.p2p import P2P, PeerID, StubBase
from hivemind.proto import runtime_pb2
from hivemind.utils import DHTExpiration, MPFuture, anext, as_aiter, get_dht_time, nested_flatten

from src.server.handler import TransformerConnectionHandler


class RemoteTransformerBlock(RemoteExpert):
    """A class that interacts with a remote module on a specific server for forward/backward or inference"""

    @property
    def stub(self) -> StubBase:
        return TransformerConnectionHandler.get_stub(self.p2p, self.peer_id)

    def begin_inference_session(self) -> RemoteTransformerBlockInferenceSession:
        """Initialize a new inference session with the specified remote server"""
        return RemoteExpertWorker.run_coroutine(RemoteTransformerBlockInferenceSession._create(self))


class RemoteTransformerBlockInferenceSession:
    """An interface to a single multi-step *inference* session for a specific remote module with a specific server"""

    def __init__(self, uid: ExpertUID, info: Dict[str, Any], inputs_queue: asyncio.Queue, outputs_aiter: AsyncIterator):
        self.uid, self.info = uid, info
        # warning: this code manages async objects that are only usable inside RemoteExpertWorker's background thread;
        # using them in any other EventLoop may cause side-effects including, headaches, diarrhea, and loss of sleep
        self._inputs_queue: asyncio.Queue[runtime_pb2.ExpertRequest] = inputs_queue
        self._outputs_stream: AsyncIterator[runtime_pb2.ExpertResponse] = outputs_aiter
        self.closed = False

    @classmethod
    async def _create(
        cls, remote_module: RemoteTransformerBlock, timeout: Optional[float] = None
    ) -> RemoteTransformerBlockInferenceSession:
        """Create a new session for a given remote module. This code is meant to be run inside RemoteExpertWorker"""
        inputs_queue = asyncio.Queue()
        outputs_stream = await remote_module.stub.rpc_inference(
            cls._read_inputs_from_queue(inputs_queue, timeout), timeout=timeout
        )
        return cls(remote_module.uid, remote_module.info, inputs_queue, outputs_stream)

    @staticmethod
    async def _read_inputs_from_queue(queue: asyncio.Queue, timeout: Optional[float]) -> AsyncIterator:
        while True:
            next_input_message = await asyncio.wait_for(queue.get(), timeout)
            yield next_input_message
            if not next_input_message.uid and not next_input_message.tensors:
                break  # this message means "done sending"

    def step(self, new_hidden_states: torch.Tensor):
        """Inference step: send a chunk of input tensors and receive a chunk of outputs"""
        if self.closed:
            raise Exception("Session is closed, cannot perform step")
        # serialize inputs and put them into the queue
        inputs = (new_hidden_states,)
        outputs_serialized = RemoteExpertWorker.run_coroutine(
            self._step(
                runtime_pb2.ExpertRequest(
                    uid=self.uid,
                    tensors=[
                        serialize_torch_tensor(tensor, proto.compression)
                        for tensor, proto in zip(inputs, nested_flatten(self.info["forward_schema"]))
                    ],
                )
            )
        )
        outputs = list(map(deserialize_torch_tensor, outputs_serialized.tensors))
        assert outputs[0].shape == inputs[0].shape, f"expected outputs[0] to be hidden states but got {outputs[0]}"
        return outputs[0]

    async def _step(self, inputs_serialized: runtime_pb2.ExpertRequest) -> runtime_pb2.ExpertResponse:
        """Inference step on serialized data. This code is meant to be run inside RemoteExpertWorker"""
        await self._inputs_queue.put(inputs_serialized)
        return await anext(self._outputs_stream)

    def close(self):
        """Finish a given inference session, close the underlying connection"""
        if self._outputs_stream is None:
            return  # already closed
        RemoteExpertWorker.run_coroutine(self._aclose_stream())
        self._outputs_stream = self._inputs_queue = None
        self.closed = True

    async def _aclose_stream(self):
        """Close the inference session. This code is meant to be run inside RemoteExpertWorker"""
        if self._outputs_stream is None:
            return  # already closed
        await self._inputs_queue.put(runtime_pb2.ExpertRequest())  # empty request will trigger end of session
        try:
            await anext(self._outputs_stream)
        except StopAsyncIteration:
            pass

    def __del__(self):
        self.close()

    def __enter__(self):
        assert not self.closed
        return self

    def __exit__(self, *exc_details):
        self.close()


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
    infos = dht.run_coroutine(
        partial(_get_remote_module_infos, uids=list(uids), expiration_time=expiration_time), return_future
    )

    if return_future:

        async def _unpack(infos_future: MPFuture, dht: DHT):
            p2p = await dht.replicate_p2p()
            return _create_remote_modules_from_infos(await infos_future, p2p)

        return RemoteExpertWorker.run_coroutine(_unpack(infos, dht), return_future)
    p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p())
    return _create_remote_modules_from_infos(infos, p2p)


async def _get_remote_module_infos(
    dht: DHT, node: DHTNode, uids: List[ExpertUID], expiration_time: Optional[DHTExpiration]
) -> List[Optional[RemoteModuleInfo]]:
    if expiration_time is None:
        expiration_time = get_dht_time()
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    found: Dict[ExpertUID, DHTValue] = await node.get_many(uids, expiration_time, num_workers=num_workers)

    experts: List[Optional[RemoteModuleInfo]] = [None] * len(uids)
    for i, uid in enumerate(uids):
        server_peer_id = found[uid]
        if server_peer_id is not None and isinstance(server_peer_id.value, str):
            experts[i] = RemoteModuleInfo(uid, PeerID.from_base58(server_peer_id.value))
    return experts


def _create_remote_modules_from_infos(
    infos: Sequence[Optional[RemoteModuleInfo]], p2p: P2P
) -> List[Optional[RemoteTransformerBlock]]:
    experts: List[Optional[RemoteTransformerBlock]] = []
    for info in infos:
        if info is not None:
            experts.append(RemoteTransformerBlock(info, p2p))
        else:
            experts.append(None)
    return experts
