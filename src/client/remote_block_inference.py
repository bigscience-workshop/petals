from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

import torch
from hivemind import serialize_torch_tensor, nested_flatten, deserialize_torch_tensor, anext
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import StubBase
from hivemind.proto import runtime_pb2

from src.data_structures import ModuleUID, RPCInfo


class RemoteTransformerBlockInferenceSession:
    """An interface to a single multi-step *inference* session for a specific remote module with a specific server"""

    def __init__(self, uid: ModuleUID, rpc_info: RPCInfo, inputs_queue: asyncio.Queue, outputs_aiter: AsyncIterator):
        self.uid, self.rpc_info = uid, rpc_info
        # warning: this code manages async objects that are only usable inside RemoteExpertWorker's background thread;
        # using them in any other EventLoop may cause side-effects including, headaches, diarrhea, and loss of sleep
        self._inputs_queue: asyncio.Queue[runtime_pb2.ExpertRequest] = inputs_queue
        self._outputs_stream: AsyncIterator[runtime_pb2.ExpertResponse] = outputs_aiter
        self.stepped = False
        self.closed = False

    @classmethod
    async def _create(
        cls, stub: StubBase, uid: ModuleUID, rpc_info: RPCInfo, timeout: Optional[float] = None
    ) -> RemoteTransformerBlockInferenceSession:
        """Create a new session for a given remote module. This code is meant to be run inside RemoteExpertWorker"""
        inputs_queue = asyncio.Queue()
        outputs_stream = await stub.rpc_inference(cls._read_inputs_from_queue(inputs_queue, timeout), timeout=timeout)
        return cls(uid, rpc_info, inputs_queue, outputs_stream)

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
                        for tensor, proto in zip(inputs, nested_flatten(self.rpc_info["forward_schema"]))
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
        self.stepped = True
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
        if self.stepped:
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
