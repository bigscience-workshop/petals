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
)
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import P2PHandlerError, StubBase
from hivemind.proto import runtime_pb2

from petals.client.routing.sequence_manager import RemoteSequenceManager, maybe_log_traceback
from petals.data_structures import CHAIN_DELIMITER, ModuleUID, RemoteSpanInfo, RPCInfo
from petals.server.handler import TransformerConnectionHandler
from petals.utils.misc import DUMMY, is_dummy

logger = get_logger(__file__)


class _ServerInferenceSession:
    """
    An interface to a single multi-step *inference* session for a a set of blocks on a specific server.

    :note: This class is *not* fault-tolerant out of the box.
    """

    def __init__(
        self,
        uid: ModuleUID,
        rpc_info: RPCInfo,
        inputs_queue: asyncio.Queue,
        outputs_aiter: AsyncIterator,
        *,
        timeout: float,
        max_length: int,
        **metadata,
    ):
        self.uid, self.rpc_info = uid, rpc_info
        self.num_blocks = uid.count(CHAIN_DELIMITER) + 1
        self._inputs_queue: asyncio.Queue[runtime_pb2.ExpertRequest] = inputs_queue
        self._outputs_stream: AsyncIterator[runtime_pb2.ExpertResponse] = outputs_aiter
        self.timeout = timeout
        self._serialized_metadata = MSGPackSerializer.dumps(dict(max_length=max_length, **metadata))
        self.stepped = False
        self.closed = False

    @classmethod
    async def create(
        cls, stub: StubBase, uid: ModuleUID, rpc_info: RPCInfo, timeout: float, **metadata
    ) -> _ServerInferenceSession:
        """Create a new session for a given remote module. This code is meant to be run inside RemoteExpertWorker"""
        inputs_queue = asyncio.Queue()
        outputs_stream = await asyncio.wait_for(
            stub.rpc_inference(cls._read_inputs_from_queue(inputs_queue)),
            timeout,
        )
        return cls(uid, rpc_info, inputs_queue, outputs_stream, timeout=timeout, **metadata)

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
    ) -> torch.Tensor:
        """
        Inference step: send a chunk of input tesors and receive a chunk of outputs
        :prompts: optional DEEP prompts, added to a prefix of each layer's outputs,
          if specified, deep prompts should have shape [num_layers, batch_size, prefix_len, hid_size]
        """
        if self.closed:
            raise Exception("Session is closed, cannot perform step")
        if prompts is None or is_dummy(prompts):
            prompts = DUMMY
        else:
            assert prompts.ndim == 4, "deep prompts should have shape [num_layers, batch_size, prefix_len, hid_size]"
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
        return await asyncio.wait_for(anext(self._outputs_stream), self.timeout)

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


class InferenceSession:
    """
    An interface to a multi-step *inference* session for a sequence of remote transformer blocks
    """

    def __init__(self, sequence_manager: RemoteSequenceManager, p2p: P2P, max_length: int):
        self._sequence_manager = sequence_manager
        self._p2p = p2p
        self._closed = False
        self._chosen_spans = []
        self._server_sessions = []
        self._server_inputs = []  # Used in case of server failures to regenerate attention caches on new servers
        self._position = 0
        self._max_length = max_length
        self.last_token_id = None

    @property
    def position(self) -> int:
        return self._position

    def _enter_server_sessions(self, chosen_spans: List[RemoteSpanInfo]) -> List[_ServerInferenceSession]:
        server_sessions = []
        try:
            for span in chosen_spans:
                stub = TransformerConnectionHandler.get_stub(self._p2p, span.peer_id)
                span_uids = CHAIN_DELIMITER.join(self._sequence_manager.block_uids[span.start : span.end])
                metadata = self._sequence_manager.get_request_metadata("rpc_inference", span_uids, peer_id=span.peer_id)
                session = RemoteExpertWorker.run_coroutine(
                    _ServerInferenceSession.create(
                        stub,
                        span_uids,
                        rpc_info=self._sequence_manager.rpc_info,
                        timeout=self._sequence_manager.request_timeout,
                        max_length=self._max_length,
                        **metadata,
                    )
                )
                server_sessions.append(session)
                session.__enter__()
            return server_sessions
        except:
            self._exit_server_sessions(server_sessions)
            raise

    def _exit_server_sessions(self, server_sessions: List[_ServerInferenceSession]) -> None:
        for session in reversed(server_sessions):
            try:
                session.__exit__(None, None, None)
            except Exception:
                logger.debug("Caught exception while closing connection to server:", exc_info=True)

    def __enter__(self) -> "InferenceSession":
        assert not self._closed and not self._chosen_spans
        return self

    def step(self, inputs: torch.Tensor, prompts: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        assert not self._closed
        if torch.is_grad_enabled():
            logger.warning("Running inference session with grad enabled. Gradients will *not* be propagated correctly.")

        n_blocks = len(self._sequence_manager)
        if prompts is None or is_dummy(prompts):
            prompts = DUMMY
        else:
            assert prompts.ndim == 4 and prompts.shape[0] == n_blocks

        inputs_device = inputs.device
        inputs_dtype = inputs.dtype
        inputs = inputs.cpu()
        prompts = prompts.cpu()

        n_input_tokens = inputs.shape[1]
        if self._position + n_input_tokens > self._max_length:
            raise ValueError(
                f"Maximum length exceeded: prefix {self._position} + current {n_input_tokens} exceeds pre-allocated maximum {self._max_length}"
            )

        server_idx = 0
        block_idx = 0
        recovery_until = -1  # Recovery mode is disabled until a failure happens
        while block_idx < n_blocks:
            for attempt_no in itertools.count():
                logger.debug(f"Inference: block {block_idx}, attempt {attempt_no}")
                span = None
                try:
                    if not self._chosen_spans or not self._server_sessions or attempt_no >= 1:
                        # If there is a failed server session, this code closes it
                        self._exit_server_sessions(self._server_sessions[server_idx : server_idx + 1])

                        n_prev_spans = len(self._chosen_spans)
                        update_end = self._chosen_spans[server_idx].end if server_idx < n_prev_spans else n_blocks
                        if attempt_no >= 1 and update_end > recovery_until:
                            logger.info(
                                f"Due to a server failure, remote attention caches "
                                f"from block {block_idx} to {update_end} will be regenerated"
                            )
                        recovery_until = max(recovery_until, update_end)

                        updated_spans = self._sequence_manager.make_sequence(block_idx, update_end)
                        # make_sequence() could return a longer sequence
                        updated_spans[-1].end = min(updated_spans[-1].end, update_end)
                        updated_sessions = self._enter_server_sessions(updated_spans)
                        logger.debug(
                            f"Found path from block {block_idx} to {update_end} via {len(updated_spans)} servers"
                        )

                        # If there is a failed span, this code replaces it, otherwise it just adds new ones
                        self._chosen_spans[server_idx : server_idx + 1] = updated_spans
                        self._server_sessions[server_idx : server_idx + 1] = updated_sessions
                        recovery_inputs = self._server_inputs[server_idx] if server_idx < n_prev_spans else None
                        self._server_inputs[server_idx : server_idx + 1] = [recovery_inputs] + [None] * (
                            len(updated_spans) - 1
                        )
                        assert len(self._chosen_spans) == len(self._server_sessions) == len(self._server_inputs), (
                            f"Broken state: {len(self._chosen_spans)} spans, {len(self._server_sessions)} sessions, "
                            f"{len(self._server_inputs)} inputs"
                        )

                    session = self._server_sessions[server_idx]
                    span = self._chosen_spans[server_idx]

                    if self._server_inputs[server_idx] is None:
                        self._server_inputs[server_idx] = inputs
                    elif self._server_inputs[server_idx].shape[1] == self._position:
                        self._server_inputs[server_idx] = torch.cat(
                            [self._server_inputs[server_idx], inputs[:, -n_input_tokens:]], dim=1
                        )
                    assert self._server_inputs[server_idx].shape[1] == self._position + n_input_tokens, (
                        f"Broken input cache: server_idx={server_idx} shape={self._server_inputs[server_idx].shape} "
                        f"position={self._position} n_input_tokens={n_input_tokens}"
                    )

                    if not session.stepped:
                        inputs = self._server_inputs[server_idx]  # Pass full inputs including prefix
                    else:
                        inputs = inputs[:, -n_input_tokens:]  # No need to pass prefix further

                    outputs = session.step(inputs, prompts[span.start : span.end], **kwargs)
                    assert (
                        inputs.shape == outputs.shape
                    ), f"Shape mismatch: inputs.shape={inputs.shape}, outputs.shape={outputs.shape})"

                    inputs = outputs
                    server_idx += 1
                    block_idx = span.end
                    self._sequence_manager.on_request_success(span.peer_id)
                    break
                except Exception as e:
                    if span is not None and not isinstance(e, P2PHandlerError):
                        self._sequence_manager.on_request_failure(span.peer_id)
                    delay = self._sequence_manager.get_retry_delay(attempt_no)
                    logger.warning(
                        f"Caught exception when running inference from block {block_idx} "
                        f"(retry in {delay:.0f} sec): {repr(e)}"
                    )
                    maybe_log_traceback(e)
                    time.sleep(delay)

        self._position += n_input_tokens
        inputs = inputs[:, -n_input_tokens:]
        outputs = inputs.to(device=inputs_device, dtype=inputs_dtype)
        return outputs

    def close(self, *exc_details):
        """Finish a given inference session, close the underlying connection"""
        if not self._closed:
            self._server_inputs.clear()
            self._exit_server_sessions(self._server_sessions)
            self._server_sessions.clear()
            self._chosen_spans.clear()
            self._closed = True

    def __exit__(self, *exc_details):
        self.close(*exc_details)

    def __del__(self):
        self.close()
