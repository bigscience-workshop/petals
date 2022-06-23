# Note: this code is being actively modified by justheuristic. If you want to change anything about it, please warn me.
from __future__ import annotations

import asyncio
import random
from typing import Any, AsyncIterator, Dict, Optional

import torch
from hivemind.compression import deserialize_torch_tensor, serialize_torch_tensor
from hivemind.moe.client.expert import RemoteExpert, RemoteExpertWorker
from hivemind.moe.expert_uid import ExpertInfo
from hivemind.p2p import P2P, StubBase
from hivemind.proto import runtime_pb2
from hivemind.utils import anext, nested_flatten

from src.data_structures import RemoteModuleInfo
from src.dht_utils import ModuleUID
from src.server.handler import TransformerConnectionHandler


class RemoteTransformerBlock(RemoteExpert):
    """A class that interacts with a remote module on a specific server for forward/backward or inference"""

    def __init__(self, peers_info: RemoteModuleInfo, p2p: P2P):
        peer_info = ExpertInfo(peers_info.uid, random.choice(list(peers_info.peer_ids)))  # TODO replace this
        super().__init__(peer_info, p2p)

    @property
    def stub(self) -> StubBase:
        return TransformerConnectionHandler.get_stub(self.p2p, self.peer_id)

    def begin_inference_session(self) -> RemoteTransformerBlockInferenceSession:
        """Initialize a new inference session with the specified remote server"""
        _ = self.info  # create _info manually since the built-in property will not work inside RemoteExpertWorker
        return RemoteExpertWorker.run_coroutine(RemoteTransformerBlockInferenceSession._create(self))


class RemoteTransformerBlockInferenceSession:
    """An interface to a single multi-step *inference* session for a specific remote module with a specific server"""

    def __init__(self, uid: ModuleUID, info: Dict[str, Any], inputs_queue: asyncio.Queue, outputs_aiter: AsyncIterator):
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
