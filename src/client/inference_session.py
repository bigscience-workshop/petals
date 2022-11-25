from __future__ import annotations

import asyncio
import itertools
import logging
import time
from typing import AsyncIterator, List, Optional

import torch
from hivemind import (
    P2P,
    MSGPackSerializer,
    anext,
    deserialize_torch_tensor,
    get_logger,
    nested_flatten,
    serialize_torch_tensor,
    use_hivemind_log_handler,
)
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import StubBase
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import aiter_with_timeout

from src.client.sequence_manager import RemoteSequenceManager
from src.data_structures import CHAIN_DELIMITER, ModuleUID, RemoteSpanInfo, RPCInfo
from src.server.handler import TransformerConnectionHandler
from src.utils.misc import DUMMY, is_dummy

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class RemoteServerInferenceSession:
    """
    An interface to a single multi-step *inference* session for a a set of blocks on a specific server.

    :note: this inference session is *not* fault-tolerant out of the box
    """

    def __init__(
        self,
        uid: ModuleUID,
        rpc_info: RPCInfo,
        inputs_queue: asyncio.Queue,
        outputs_aiter: AsyncIterator,
        *,
        max_length: int,
        points: int = 0,
    ):
        self.uid, self.rpc_info = uid, rpc_info
        self.num_blocks = uid.count(CHAIN_DELIMITER) + 1
        # warning: this code manages async objects that are only usable inside RemoteExpertWorker's background thread;
        # using them in any other EventLoop may cause side-effects including, headaches, diarrhea, and loss of sleep
        self._inputs_queue: asyncio.Queue[runtime_pb2.ExpertRequest] = inputs_queue
        self._outputs_stream: AsyncIterator[runtime_pb2.ExpertResponse] = outputs_aiter
        self._serialized_metadata = MSGPackSerializer.dumps(dict(max_length=max_length, points=points))
        self.stepped = False
        self.closed = False

    @classmethod
    async def create(
        cls, stub: StubBase, uid: ModuleUID, rpc_info: RPCInfo, timeout: float, **metadata
    ) -> RemoteServerInferenceSession:
        """Create a new session for a given remote module. This code is meant to be run inside RemoteExpertWorker"""
        inputs_queue = asyncio.Queue()
        outputs_stream = await asyncio.wait_for(
            stub.rpc_inference(cls._read_inputs_from_queue(inputs_queue)),
            timeout,
        )
        outputs_stream = aiter_with_timeout(outputs_stream, timeout)
        return cls(uid, rpc_info, inputs_queue, outputs_stream, **metadata)

    @staticmethod
    async def _read_inputs_from_queue(queue: asyncio.Queue, input_timeout: Optional[float] = None) -> AsyncIterator:
        while True:
            next_input_message = await asyncio.wait_for(queue.get(), input_timeout)
            yield next_input_message
            if not next_input_message.uid and not next_input_message.tensors:
                break  # this message means "done sending"

    def step(
        self,
        new_hidden_states: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        hypo_ids: Optional[torch.Tensor] = None,
    ):
        """
        Inference step: send a chunk of input tesors and receive a chunk of outputs
        :prompts: optional DEEP prompts, added to a prefix of each layer's outputs,
          if specified, deep promts should have shape [num_layers, batch_size, prefix_len, hid_size]
        """
        if self.closed:
            raise Exception("Session is closed, cannot perform step")
        if prompts is None or is_dummy(prompts):
            prompts = DUMMY
        else:
            assert prompts.ndim == 4, "deep promts should have shape [num_layers, batch_size, prefix_len, hid_size]"
            assert prompts.shape[0] == self.num_blocks
            assert prompts.shape[1] in (new_hidden_states.shape[0], 1)
            assert prompts.shape[2] <= new_hidden_states.shape[1]
            assert prompts.shape[3] == new_hidden_states.shape[2]

        if hypo_ids is None or is_dummy(hypo_ids):
            hypo_ids = DUMMY
        else:
            assert len(hypo_ids) == len(new_hidden_states)
            assert hypo_ids.dtype == torch.int64

        # serialize inputs and put them into the queue
        inputs = (new_hidden_states, prompts, hypo_ids)
        outputs_serialized = RemoteExpertWorker.run_coroutine(
            self._step(
                runtime_pb2.ExpertRequest(
                    uid=self.uid,
                    tensors=[
                        serialize_torch_tensor(tensor.to(proto.dtype), proto.compression)
                        for tensor, proto in zip(inputs, nested_flatten(self.rpc_info["inference_schema"]))
                    ],
                    metadata=self._serialized_metadata if not self.stepped else None,
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


class RemoteSequentialInferenceSession:
    """
    An interface to a multi-step *inference* session for a sequence of remote transformer blocks
    """

    def __init__(self, sequence_manager: RemoteSequenceManager, p2p: P2P, **metadata):
        self.sequence_manager = sequence_manager
        self.p2p = p2p
        self.closed = False
        self.chosen_spans = []
        self.server_sessions = []
        self.metadata = metadata

    def _enter_server_sessions(self, chosen_spans: List[RemoteSpanInfo]) -> List[RemoteServerInferenceSession]:
        server_sessions = []
        for span in chosen_spans:
            stub = TransformerConnectionHandler.get_stub(self.p2p, span.peer_id)
            span_uids = CHAIN_DELIMITER.join(self.sequence_manager.block_uids[span.start : span.end])
            session = RemoteExpertWorker.run_coroutine(
                RemoteServerInferenceSession.create(
                    stub, span_uids, rpc_info=self.sequence_manager.rpc_info, timeout=self.sequence_manager.timeout,
                    **self.metadata
                )
            )
            server_sessions.append(session)
            session.__enter__()
        return server_sessions

    def _exit_server_sessions(self, server_sessions: List[RemoteServerInferenceSession], *, verbose: bool) -> None:
        exc_loglevel = logging.WARNING if verbose else logging.DEBUG
        for session in reversed(server_sessions):
            try:
                session.__exit__(None, None, None)
            except Exception:
                logger.log(exc_loglevel, "Caught exception while closing connection to server:", exc_info=True)

    def __enter__(self):
        assert not self.closed and not self.chosen_spans
        return self

    def step(self, inputs: torch.Tensor, prompts: Optional[torch.Tensor] = None, **kwargs):
        assert not self.closed
        if torch.is_grad_enabled():
            logger.warning("Running inference session with grad enabled. Gradients will *not* be propagated correctly.")
        if prompts is None or is_dummy(prompts):
            prompts = DUMMY
        else:
            assert prompts.ndim == 4 and prompts.shape[0] == len(self.sequence_manager)

        server_idx = 0
        block_idx = 0
        while block_idx < len(self.sequence_manager):
            for attempt_no in itertools.count():
                logger.debug(f"Inference: block {block_idx}, attempt {attempt_no}")
                try:
                    if not self.chosen_spans or not self.server_sessions or attempt_no >= 1:
                        self._exit_server_sessions(self.server_sessions[server_idx:], verbose=False)
                        self.server_sessions[server_idx:] = []
                        self.chosen_spans[server_idx:] = []

                        self.sequence_manager.update_()
                        backup_spans = self.sequence_manager.make_sequence(block_idx)
                        self.chosen_spans.extend(backup_spans)
                        self.server_sessions.extend(self._enter_server_sessions(backup_spans))
                        logger.debug(f"Found path from block {block_idx} via {len(backup_spans)} servers")

                    session = self.server_sessions[server_idx]
                    span = self.chosen_spans[server_idx]

                    outputs = session.step(inputs, prompts[span.start : span.end], **kwargs)
                    assert outputs.shape == inputs.shape, f"expected {inputs.shape}, got {outputs.shape}"
                    inputs = outputs

                    server_idx += 1
                    block_idx = span.end
                    break
                except Exception as e:
                    delay = self.sequence_manager.min_backoff * 2**attempt_no
                    logger.warning(
                        f"Caught exception when running inference from block {block_idx} "
                        f"(retry in {delay:.2f} sec): {repr(e)}"
                    )
                    logger.debug("See detailed traceback below:", exc_info=True)
                    time.sleep(delay)
        return inputs

    def close(self, *exc_details):
        """Finish a given inference session, close the underlying connection"""
        if not self.closed:
            self._exit_server_sessions(self.server_sessions, verbose=True)
            self.server_sessions.clear()
            self.closed = True

    def __exit__(self, *exc_details):
        self.close(*exc_details)

    def __del__(self):
        self.close()
