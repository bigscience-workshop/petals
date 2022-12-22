import asyncio
import contextlib
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from async_timeout import timeout
from hivemind import (
    DHT,
    MSGPackSerializer,
    P2PContext,
    TensorDescriptor,
    deserialize_tensor_stream,
    deserialize_torch_tensor,
    nested_flatten,
    serialize_torch_tensor,
)
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import amap_in_executor, anext
from hivemind.utils.logging import get_logger
from hivemind.utils.streaming import split_for_streaming

from petals.data_structures import CHAIN_DELIMITER, ModuleUID
from petals.server.backend import TransformerBackend
from petals.server.task_pool import PrioritizedTaskPool
from petals.server.task_prioritizer import DummyTaskPrioritizer, TaskPrioritizerBase
from petals.utils.misc import DUMMY, is_dummy

logger = get_logger(__file__)


class TransformerConnectionHandler(ConnectionHandler):
    """Handles three request types: forward, backward and forward-incremental (inference)"""

    module_backends: Dict[ModuleUID, TransformerBackend]

    def __init__(
        self,
        dht: DHT,
        module_backends: Dict[str, TransformerBackend],
        *,
        inference_max_length: int,
        request_timeout: float,
        session_timeout: float,
        step_timeout: float,
        task_prioritizer: TaskPrioritizerBase = DummyTaskPrioritizer(),
    ):
        super().__init__(dht, module_backends)
        for module_backend in self.module_backends.values():
            assert isinstance(module_backend, TransformerBackend)
        self.inference_max_length = inference_max_length
        self.request_timeout = request_timeout
        self.session_timeout, self.step_timeout = session_timeout, step_timeout
        self._prioritizer = task_prioritizer

    def shutdown(self):
        if self.is_alive():
            self._outer_pipe.send("_shutdown")
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning(f"{self.__class__.__name__} failed to shut down gracefully, sending SIGTERM")
                self.terminate()

    async def _gather_inputs(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> Tuple[str, List[torch.Tensor], Dict]:
        block_uid, metadata = None, None

        def _unpack(req: runtime_pb2.ExpertRequest) -> Iterable[runtime_pb2.Tensor]:
            nonlocal block_uid, metadata

            if block_uid is None:
                block_uid = req.uid
            elif block_uid != req.uid:
                raise ValueError("Block uids differ in one request")

            if metadata is None:
                metadata = MSGPackSerializer.loads(req.metadata) if req.metadata else {}

            return req.tensors

        tensors_stream = amap_in_executor(_unpack, requests)
        inputs = await deserialize_tensor_stream(tensors_stream)
        assert isinstance(block_uid, str) and isinstance(metadata, dict)
        return block_uid, inputs, metadata

    async def rpc_inference(
        self,
        requests: AsyncIterator[runtime_pb2.ExpertRequest],
        context: P2PContext,
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:
        """Compute a single step of inference using attention cache; update attention cache accordingly."""

        async with timeout(self.session_timeout):
            try:
                request = await asyncio.wait_for(anext(requests), self.step_timeout)
            except asyncio.TimeoutError:
                self._log_request("rpc_inference.open", None, context, warning="timed out")
                return

            requested_uids = self._check_uids(request.uid)
            self._log_request("rpc_inference.open", requested_uids, context)
            try:
                metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
                requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
                max_length = metadata.get("max_length")
                points = metadata.get("points", 0)

                if not requested_uids:
                    raise ValueError("User must specify at least one block for inference, but got none")
                assert isinstance(
                    max_length, int
                ), f"rpc_inference metadata must contain int max_length, got {max_length}"
                assert isinstance(
                    points, (float, int)
                ), f"rpc_inference should have number of points as a number or None, got {points}"
                if not 0 <= max_length <= self.inference_max_length:
                    raise ValueError(
                        f"Cannot allocate KV cache for {max_length} tokens, max = {self.inference_max_length}"
                    )

                point_per_piece = points / max_length if max_length > 0 else 0.0
                batch_size = request.tensors[0].size[0] if request.tensors else 1

                cache_metadata = torch.tensor(
                    [[-1, -1, -1] for _ in range(batch_size)], dtype=torch.int64
                )  # [cache_handle, rel_index, prefix_length]
                prefix_length = 0

                async with self._allocate_cache(requested_backends, batch_size, max_length) as cache_handle:
                    while request.tensors:  # iterate while user is willing to supply tensors
                        hidden_states, prompts, hypo_ids = [
                            deserialize_torch_tensor(tensor) for tensor in request.tensors
                        ]

                        # Cast inputs to backend dtype
                        hidden_states = hidden_states.to(requested_backends[0].dtype)
                        assert hypo_ids.dtype == torch.int64, f"hypo ids must be int64, got {hypo_ids.dtype}"

                        # parse deep prompts (optional argument)
                        if prompts is None or is_dummy(prompts) or is_dummy(prompts):
                            prompts = [DUMMY] * len(requested_backends)
                        else:
                            prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]

                        if not (len(requested_backends) == len(prompts)):
                            raise ValueError(f"Received {len(prompts)} prompts for {len(requested_backends)} backends")

                        length_increment = hidden_states.shape[1]  # how many tokens are added this step (in each seq)
                        if prefix_length + length_increment > max_length:
                            raise ValueError(
                                f"Maximum length exceeded: prefix {prefix_length} + current {length_increment}"
                                f" exceeds pre-allocated maximum {max_length}"
                            )

                        # run request tensors through all requested modules, update caches
                        for rel_index, (backend, prompt) in enumerate(zip(requested_backends, prompts)):
                            if not is_dummy(prompt):
                                hidden_states[:, : prompt.shape[1]] += prompt
                            if hidden_states.numel() == 0:
                                continue  # user passed a tensor with 0 tokens. This is a special case that occurs, e.g.
                                # when user wants to pre-allocate cache or check that server *can* allocate that cache

                            cache_metadata[:] = torch.tensor(
                                [cache_handle, rel_index, prefix_length], dtype=torch.int64
                            )
                            assert isinstance(
                                hidden_states, torch.Tensor
                            ), f"hidden states must be tensor, got {type(hidden_states)}"
                            assert (
                                hidden_states.ndim == 3
                            ), f"inputs to {type(backend)} must be a list with a single 3d tensor of hidden states"
                            assert isinstance(
                                backend.inference_pool, PrioritizedTaskPool
                            ), "petals support only prioritized pools"
                            priority = self._prioritizer.prioritize(
                                cache_metadata,
                                hidden_states,
                                hypo_ids,
                                points=point_per_piece / len(requested_backends),
                                backend=backend,
                                type="inference",
                            )
                            (hidden_states,) = await backend.inference_pool.submit_task(
                                hidden_states, hypo_ids, cache_metadata, priority=priority
                            )

                        # serialize and send last layer outputs
                        yield runtime_pb2.ExpertResponse(
                            tensors=[
                                serialize_torch_tensor(result.to(proto.dtype), proto.compression, allow_inplace=True)
                                for result, proto in zip(
                                    (hidden_states,), nested_flatten(requested_backends[-1].outputs_schema)
                                )
                            ]
                        )

                        # prepare for next step
                        prefix_length += hidden_states.shape[1]
                        try:
                            request = await asyncio.wait_for(anext(requests), self.step_timeout)
                        except asyncio.TimeoutError:
                            self._log_request("rpc_inference.step", requested_uids, context, warning="timed out")
                            return
            finally:
                self._log_request("rpc_inference.close", requested_uids, context)

    async def rpc_forward(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        async with timeout(self.request_timeout):
            # Parse request and prepare backends
            flat_inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
            requested_uids = self._check_uids(request.uid)
            self._log_request("rpc_forward", requested_uids, context)

            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
            points = metadata.get("points", 0)
            assert isinstance(
                points, (float, int)
            ), f"rpc_forward should have number of points as number or None, got {points}"

            hidden_states = await _rpc_forward(
                *flat_inputs, requested_backends=requested_backends, prioritizer=self._prioritizer, points=points
            )
            return runtime_pb2.ExpertResponse(
                tensors=self._serialize_outputs(hidden_states, requested_backends, metadata)
            )

    async def rpc_forward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:
        async with timeout(self.request_timeout):
            # Parse requests and prepare backends
            uid_str, flat_inputs, metadata = await self._gather_inputs(requests, context)
            requested_uids = self._check_uids(uid_str)
            self._log_request("rpc_forward_stream", requested_uids, context)

            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            points = metadata.get("points", 0)
            assert isinstance(
                points, (float, int)
            ), f"rpc_forward_stream should have number of points as number or None, got {points}"

            hidden_states = await _rpc_forward(
                *flat_inputs, requested_backends=requested_backends, prioritizer=self._prioritizer, points=points
            )

            # Split the serialized_output for streaming and respond to client
            for tensor in self._serialize_outputs(hidden_states, requested_backends, metadata):
                for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE):
                    yield runtime_pb2.ExpertResponse(tensors=[part])

    def _serialize_outputs(
        self,
        hidden_states: torch.Tensor,
        requested_backends: Sequence[TransformerBackend],
        metadata: Dict[str, Any],
    ) -> Sequence[runtime_pb2.Tensor]:
        """Serialize forward outputs using either outputs_schema or custom user-specified schema"""
        assert isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3, "hidden_states must be a 3d tensor"
        outputs_schema = requested_backends[-1].outputs_schema

        if metadata.get("output_compression") is not None:
            assert isinstance(metadata["output_compression"], (list, tuple)), "output_compression must be a tuple/list"
            output_compression = tuple(metadata["output_compression"])
            assert all(isinstance(c, int) for c in output_compression), "output_compression must contain integers"
            assert len(output_compression) == 1, f"output_compression tuple should have 1 element"
        else:
            output_compression = tuple(tensor.compression for tensor in outputs_schema)

        return [
            serialize_torch_tensor(result.to(proto.dtype), compression, allow_inplace=True)
            for result, proto, compression in zip([hidden_states], outputs_schema, output_compression)
        ]

    async def rpc_backward(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        async with timeout(self.request_timeout):
            # Parse requests and prepare backends
            flat_tensors = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
            requested_uids = self._check_uids(request.uid)
            self._log_request("rpc_backward", requested_uids, context)

            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
            points = metadata.get("points", 0)
            assert isinstance(
                points, (float, int)
            ), f"rpc_backward should have number of points as number or None, got {points}"

            grads = await _rpc_backward(
                *flat_tensors, requested_backends=requested_backends, prioritizer=self._prioritizer, points=points
            )

            return runtime_pb2.ExpertResponse(tensors=self._serialize_grads(grads, requested_backends, metadata))

    async def rpc_backward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:
        async with timeout(self.request_timeout):
            uids_header, flat_tensors, metadata = await self._gather_inputs(requests, context)
            requested_uids = self._check_uids(uids_header)
            self._log_request("rpc_backward_stream", requested_uids, context)

            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)
            points = metadata.get("points", 0)
            assert isinstance(
                points, (float, int)
            ), f"rpc_backward_stream should have number of points as number or None, got {points}"

            grads = await _rpc_backward(
                *flat_tensors, requested_backends=requested_backends, prioritizer=self._prioritizer, points=points
            )
            # Split the serialized_grad_inputs for streaming and respond
            for tensor in self._serialize_grads(grads, requested_backends, metadata):
                for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE):
                    yield runtime_pb2.ExpertResponse(tensors=[part])

    def _serialize_grads(
        self,
        grads: Sequence[torch.Tensor],
        requested_backends: Sequence[TransformerBackend],
        metadata: Dict[str, Any],
    ) -> Sequence[runtime_pb2.Tensor]:
        """Serialize backward gradients w.r.t. inputs using either default schema or custom user-specified schema"""
        # Modify grad_inputs_schema to support grad_prompts
        assert len(requested_backends[0].args_schema) == 1 and len(grads) in (1, 2)  # TODO generalize
        flat_grads_schema = tuple(
            nested_flatten((requested_backends[0].args_schema * len(grads), requested_backends[0].kwargs_schema))
        )  # TODO generalize

        if metadata.get("output_compression") is not None:
            assert isinstance(metadata["output_compression"], (list, tuple)), "output_compression must be a tuple/list"
            output_compression = tuple(metadata["output_compression"])
            assert all(isinstance(c, int) for c in output_compression), "output_compression must contain integers"
            assert len(output_compression) == len(grads), f"output_compression should have {len(grads)} elements"
        else:
            output_compression = tuple(tensor.compression for tensor in flat_grads_schema)

        return [
            serialize_torch_tensor(result.to(proto.dtype), compression, allow_inplace=True)
            for result, proto, compression in zip(grads, flat_grads_schema, output_compression)
        ]

    def _check_uids(self, uids: str) -> Tuple[ModuleUID, ...]:
        """Check that the first request to rpc_inference is valid"""
        uids = (uids or "").split(CHAIN_DELIMITER)
        if not uids:
            raise RuntimeError("User did not provide any uids")
        for uid in uids:
            if uid not in self.module_backends:
                raise RuntimeError(f"Remote peer does not serve {uid}")
        return tuple(uids)

    @contextlib.asynccontextmanager
    async def _allocate_cache(
        self, backends: Sequence[TransformerBackend], batch_size: int, max_length: int
    ) -> Sequence[int]:
        """Allocate memory cache for all transformer blocks, return cache handle"""

        n_blocks = len(backends)
        backend = backends[0]
        n_heads = backend.module.self_attention.num_heads
        head_dim = backend.module.self_attention.head_dim
        descr = TensorDescriptor(size=(n_blocks, 2, batch_size, n_heads * head_dim * max_length), dtype=backend.dtype)
        alloc_size = descr.numel() * torch.finfo(descr.dtype).bits // 8

        gib = 1024**3
        cur_size = backend.memory_cache.current_size_bytes
        max_size = backend.memory_cache.max_size_bytes
        friendly_max_size = f"{max_size / gib:.2f}" if max_size != 2**64 - 1 else "inf"
        logger.info(
            f"rpc_inference.wait_for_alloc(size={alloc_size / gib:.2f} GiB), "
            f"already used {cur_size / gib:.2f}/{friendly_max_size} GiB ({cur_size / max_size * 100:.1f}%)"
        )

        async with backend.memory_cache.allocate_cache(descr) as handle:
            logger.info(f"rpc_inference.alloc(size={alloc_size / gib:.2f} GiB)")
            yield handle

    def _log_request(
        self, method: str, uids: Optional[Sequence[ModuleUID]], context: P2PContext, *, warning: Optional[str] = None
    ) -> None:
        if uids is not None:
            friendly_uids = [uid.split(".")[-1] for uid in uids if "." in uid]
            friendly_uids = [int(uid) for uid in friendly_uids if uid.isdigit()]
            friendly_uids = f"{min(friendly_uids)}:{max(friendly_uids) + 1}" if friendly_uids else uids
        else:
            friendly_uids = "n/a"

        friendly_remote_id = "..." + str(context.remote_id)[-6:]

        message = f"{method}(blocks={friendly_uids}, remote_peer={friendly_remote_id})"
        if warning is None:
            logger.info(message)
        else:
            logger.warning(f"{message}: {warning}")


async def _rpc_forward(
    *flat_tensors: torch.Tensor,
    requested_backends: Sequence[TransformerBackend],
    prioritizer: TaskPrioritizerBase,
    points: int = 0,
) -> torch.Tensor:
    """
    Run forward pass on deserialized inputs and prompts, used by rpc_forward and rpc_forward_stream

    :param flat_tensors: a list of tensors that includes first layer inputs, optional prompts and extra tensors
    :note: some input tensors can be missing, in which case they will be replaced with dummy tensors (see is_dummy)
    :param requested_backends: a sequence of transformer blocks in the same order as they appear in forward pass
    :returns: hidden states after the last layer [batch_size, seq_length, hid_size]
    """
    hidden_states, prompts = flat_tensors
    dtype = requested_backends[0].dtype
    # check parse input tensors and cast dtypes
    hidden_states = hidden_states.to(dtype)
    assert hidden_states.ndim == 3
    if prompts is None or is_dummy(prompts):
        prompts = [DUMMY] * len(requested_backends)
    else:
        prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]

    # Run a chain of requested backends
    for backend, prompt in zip(requested_backends, prompts):
        if not is_dummy(prompt):
            hidden_states[:, : prompt.shape[1]] += prompt

        assert isinstance(backend.inference_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            hidden_states, points=points / len(requested_backends), backend=backend, type="forward"
        )
        (hidden_states,) = await backend.forward_pool.submit_task(
            hidden_states,
            priority=priority,
        )
        assert isinstance(hidden_states, torch.Tensor)
        assert (
            hidden_states.ndim == 3
        ), f"inputs to {type(backend)} must be a list with a single 3d tensor of hidden states"

    # Serialize the overall output
    return hidden_states


async def _rpc_backward(
    *flat_tensors: torch.Tensor,
    requested_backends: Sequence[TransformerBackend],
    prioritizer: TaskPrioritizerBase,
    points: int = 0,
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    inputs, grad_outputs, prompts = flat_tensors
    # Cast inputs & grad outputs to backend dtype
    inputs = inputs.to(requested_backends[0].dtype)
    grad_outputs = grad_outputs.to(requested_backends[-1].dtype)

    if prompts is None or is_dummy(prompts):
        prompts = [DUMMY] * len(requested_backends)
    else:
        prompts = [p.squeeze(0) for p in prompts.to(requested_backends[0].dtype).split(1, dim=0)]

    # Run a forward chain to collect intermediate inputs
    # Note that we do not forward for the last module since we do not need its output
    inter_inputs = []
    for backend, prompt in zip(requested_backends[:-1], prompts[:-1]):
        assert inputs.ndim == 3, f"inputs to {type(backend)} must be a single 3d tensor of hidden states"
        if not is_dummy(prompt):
            inputs[:, : prompt.shape[1]] += prompt
        inter_inputs.append(inputs)
        assert isinstance(backend.inference_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            inputs, points=points / len(requested_backends), backend=backend, type="forward_in_backward"
        )
        (inputs,) = await backend.forward_pool.submit_task(inputs, priority=priority)

        assert isinstance(inputs, torch.Tensor)

    if not is_dummy(prompts[-1]):
        inputs[:, : prompts[-1].shape[1]] += prompts[-1]
    inter_inputs.append(inputs)

    assert len(inter_inputs) == len(prompts) == len(requested_backends), "internal shape error during backward"
    grad_prompts_reversed = []
    # Run a chain of requested backends
    for inp, prompt, backend in zip(*map(reversed, (inter_inputs, prompts, requested_backends))):
        assert isinstance(backend.inference_pool, PrioritizedTaskPool), "petals support only prioritized pools"
        priority = prioritizer.prioritize(
            inp, grad_outputs, points=points / len(requested_backends), backend=backend, type="backward"
        )
        (grad_outputs,) = await backend.backward_pool.submit_task(inp, grad_outputs, priority=priority)

        assert isinstance(grad_outputs, torch.Tensor)
        if not is_dummy(prompt):
            grad_prompts_reversed.append(grad_outputs[:, : prompt.shape[1]].unsqueeze(0))

    grad_prompts = torch.cat(grad_prompts_reversed[::-1], dim=0) if grad_prompts_reversed else DUMMY
    return [grad_outputs] if is_dummy(grad_prompts) else [grad_outputs, grad_prompts]  # TODO un-duct-tape
