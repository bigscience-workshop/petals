import contextlib
from typing import AsyncIterator, Dict, Sequence

import torch
from hivemind import DHT, P2PContext, TensorDescriptor, deserialize_torch_tensor, nested_flatten, serialize_torch_tensor
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils import as_aiter
from hivemind.utils.asyncio import anext
from hivemind.utils.streaming import split_for_streaming

from src.data_structures import CHAIN_DELIMITER, ModuleUID
from src.server.backend import MAX_LENGTH, TransformerBackend
from src.utils.misc import DUMMY, is_dummy, is_dummy_batch, make_dummy_batch


class TransformerConnectionHandler(ConnectionHandler):
    """Handles three request types: forward, backward and forward-incremental (inference)"""

    module_backends: Dict[ModuleUID, TransformerBackend]

    def __init__(self, dht: DHT, module_backends: Dict[str, TransformerBackend]):
        super().__init__(dht, module_backends)
        for module_backend in self.module_backends.values():
            assert isinstance(module_backend, TransformerBackend)

    async def rpc_inference(
        self,
        requests: AsyncIterator[runtime_pb2.ExpertRequest],
        context: P2PContext,
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:
        """Compute a single step of inference using attention cache; update attention cache accordingly."""
        try:
            print("OPENED RPC_INFERENCE")
            request = await anext(requests)
            requested_uids = self._check_header(request)
            requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)

            batch_size = request.tensors[0].size[0] if request.tensors else 1

            cache_metadata = torch.tensor(
                [[-1, -1] for _ in range(batch_size)], dtype=torch.int64
            )  # [cache_handle, prefix_length]
            prefix_length = 0

            async with self._allocate_caches(requested_backends, batch_size) as cache_handles:
                assert len(cache_handles) == len(requested_backends)
                while request.tensors:  # iterate while user is willing to supply tensors
                    hidden_states = [deserialize_torch_tensor(tensor) for tensor in request.tensors]

                    # Cast inputs to backend dtype
                    hidden_states = [tensor.to(requested_backends[0].dtype) for tensor in hidden_states]

                    # run request tensors through all requested modules, update caches
                    for backend, cache_handle in zip(requested_backends, cache_handles):
                        cache_metadata[:, 0], cache_metadata[:, 1] = cache_handle, prefix_length
                        assert (
                            len(hidden_states) == 1 and hidden_states[0].ndim == 3
                        ), f"inputs to {type(backend)} must be a list with a single 3d tensor of hidden states"

                        hidden_states = await backend.inference_pool.submit_task(cache_metadata, *hidden_states)
                        assert isinstance(hidden_states, (list, tuple))
                        assert len(hidden_states) == 1 and hidden_states[0].ndim == 3

                    # serialize and send last layer outputs
                    yield runtime_pb2.ExpertResponse(
                        tensors=[
                            serialize_torch_tensor(result.to(proto.dtype), proto.compression, allow_inplace=True)
                            for result, proto in zip(
                                hidden_states, nested_flatten(requested_backends[-1].outputs_schema)
                            )
                        ]
                    )

                    # prepare for next step
                    prefix_length += hidden_states[0].shape[1]
                    request = await (anext(requests))
        finally:
            print("CLOSED RPC_INFERENCE")

    async def rpc_forward(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        # Parse request and prepare backends
        inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        requested_uids = self._check_header(request)
        requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)

        hidden_states = await _rpc_forward(inputs, requested_backends)

        # Serialize the overall output and respond
        assert len(hidden_states) == 1 and hidden_states[0].ndim == 3
        return runtime_pb2.ExpertResponse(
            tensors=[
                serialize_torch_tensor(result.to(proto.dtype), proto.compression, allow_inplace=True)
                for result, proto in zip(hidden_states, nested_flatten(requested_backends[-1].outputs_schema))
            ]
        )

    async def rpc_forward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertRequest]:
        # Parse requests and prepare backends
        uids_header, inputs = await self._gather_inputs(requests, context)
        requested_uids = self._check_header_str(uids_header)
        requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)

        hidden_states = await _rpc_forward(inputs, requested_backends)

        # Serialize the overall output
        assert len(hidden_states) == 1 and hidden_states[0].ndim == 3
        serialized_output = [
            serialize_torch_tensor(result.to(proto.dtype), proto.compression, allow_inplace=True)
            for result, proto in zip(hidden_states, nested_flatten(requested_backends[-1].outputs_schema))
        ]

        # Split the serialized_output for streaming and respond
        output_split = [
            part for tensor in serialized_output for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
        ]
        async for part in as_aiter(*output_split):
            yield runtime_pb2.ExpertResponse(tensors=[part])

    async def rpc_backward(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        # Parse requests and prepare backends
        inputs, prompts, grad_outputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        requested_uids = self._check_header(request)
        requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)

        grads = await _rpc_backward(inputs, prompts, grad_outputs, requested_backends)

        # Serialize the overall grad_input and respond
        return runtime_pb2.ExpertResponse(
            tensors=[
                serialize_torch_tensor(result.to(proto.dtype), proto.compression, allow_inplace=True)
                for result, proto in zip(grads, nested_flatten(requested_backends[0].grad_inputs_schema))
            ]
        )

    async def rpc_backward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:

        uids_header, (inputs, prompts, grad_outputs) = await self._gather_inputs(requests, context)
        requested_uids = self._check_header_str(uids_header)
        requested_backends = tuple(self.module_backends[uid] for uid in requested_uids)

        grads = await _rpc_backward(inputs, prompts, grad_outputs, requested_backends)

        # Serialize the overall grad_inputs
        serialized_grad_inputs = [
            serialize_torch_tensor(result.to(proto.dtype), proto.compression, allow_inplace=True)
            for result, proto in zip(grads, nested_flatten(requested_backends[0].grad_inputs_schema))
        ]
        # Split the serialized_grad_inputs for streaming and respond
        output_split = [
            part for tensor in serialized_grad_inputs for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
        ]

        async for part in as_aiter(*output_split):
            yield runtime_pb2.ExpertResponse(tensors=[part])

    def _check_header(self, request: runtime_pb2.ExpertRequest) -> Sequence[ModuleUID]:
        """Check that the first request to rpc_inference is valid"""
        uids = (request.uid or "").split(CHAIN_DELIMITER)
        if not uids:
            raise RuntimeError("User did not provide any uids")
        for uid in uids:
            if uid not in self.module_backends:
                raise RuntimeError(f"Remote peer does not serve {uid}")
        return tuple(uids)

    def _check_header_str(self, header) -> Sequence[ModuleUID]:
        """Check that the first request to rpc_inference is valid"""
        uids = (header or "").split(CHAIN_DELIMITER)
        if not uids:
            raise RuntimeError("User did not provide any uids")
        for uid in uids:
            if uid not in self.module_backends:
                raise RuntimeError(f"Remote peer does not serve {uid}")
        return tuple(uids)

    @contextlib.asynccontextmanager
    async def _allocate_caches(self, backends: Sequence[TransformerBackend], batch_size: int) -> Sequence[int]:
        """Allocate memory caches for each transformer block, return cache handles"""
        async with contextlib.AsyncExitStack() as stack:
            handles = []
            for backend in backends:
                num_heads = backend.module.self_attention.num_heads
                head_dim = backend.module.self_attention.head_dim

                cache_descriptor = TensorDescriptor(
                    size=(2, batch_size, MAX_LENGTH, num_heads, head_dim), dtype=backend.dtype
                )
                # [key_or_value, batch_size, max_length, num_heads, head_dim]

                handles.append(await stack.enter_async_context(backend.memory_cache.allocate_cache(cache_descriptor)))

            yield handles


async def _rpc_forward(inputs, requested_backends):
    # Cast inputs to backend dtype
    hidden_states = [tensor.to(requested_backends[0].dtype) for tensor in inputs]
    assert len(hidden_states) == 2 and hidden_states[0].ndim == 3
    hidden_states, prompts = hidden_states

    if is_dummy(prompts):
        batch_size = hidden_states.shape[0]
        prompts = [make_dummy_batch(batch_size)] * len(requested_backends)

    # Run a chain of requested backends
    for backend, prompt in zip(requested_backends, prompts):
        (hidden_states,) = await backend.forward_pool.submit_task(hidden_states, prompt)
        assert isinstance(hidden_states, torch.Tensor)
        assert (
            hidden_states.ndim == 3
        ), f"inputs to {type(backend)} must be a list with a single 3d tensor of hidden states"

    # Serialize the overall output
    return [hidden_states]


async def _rpc_backward(inputs, prompts, grad_outputs, requested_backends):
    # Cast inputs & grad outputs to backend dtype
    inputs = inputs.to(requested_backends[0].dtype)
    prompts = prompts.to(requested_backends[0].dtype)
    grad_outputs = grad_outputs.to(requested_backends[-1].dtype)
    batch_size = inputs.shape[0]

    if is_dummy(prompts):
        prompts = [make_dummy_batch(batch_size)] * len(requested_backends)
    else:
        prompts = [p.squeeze(0) for p in prompts.split(1)]

    # Run a forward chain to collect intermediate inputs
    # Note that we do not forward for the last module since we do not need its output
    inter_inputs = [inputs]
    for backend, prompt in zip(requested_backends[:-1], prompts[:-1]):
        assert inputs.ndim == 3, f"inputs to {type(backend)} must be a single 3d tensor of hidden states"
        (inputs,) = await backend.forward_pool.submit_task(inputs, prompt)
        assert isinstance(inputs, torch.Tensor)
        inter_inputs.append(inputs)

    grad_prompts = []
    # Run a chain of requested backends
    for inp, prompt, backend in zip(inter_inputs[::-1], prompts[::-1], requested_backends[::-1]):
        grads = await backend.backward_pool.submit_task(inp, prompt, grad_outputs)
        assert isinstance(grads, (list, tuple)) and len(grads) == 2
        grad_outputs, grad_prompt = grads
        grad_prompts.append(grad_prompt[None])

    is_dummy_grad_prompts = [is_dummy_batch(grad_param, batch_size) for grad_param in grad_prompts]
    grad_prompts = torch.cat(grad_prompts, dim=0) if not any(is_dummy_grad_prompts) else DUMMY
    grads = [grad_outputs, grad_prompts]
    return grads
