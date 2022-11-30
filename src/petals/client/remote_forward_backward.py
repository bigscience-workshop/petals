"""
Utility functions that call RPC forward or backward on a single remote server
"""
import asyncio
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from hivemind import nested_compare, nested_flatten, nested_pack, serialize_torch_tensor
from hivemind.compression.serialization import deserialize_tensor_stream, deserialize_torch_tensor
from hivemind.p2p import StubBase
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, MAX_UNARY_PAYLOAD_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import aiter_with_timeout, iter_as_aiter
from hivemind.utils.streaming import split_for_streaming

from petals.data_structures import ModuleUID, RPCInfo


async def _forward_unary(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, timeout: float, **kwargs
) -> List[torch.Tensor]:
    outputs: runtime_pb2.ExpertResponse = await stub.rpc_forward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors), **kwargs),
        timeout=timeout,
    )
    return [deserialize_torch_tensor(t) for t in outputs.tensors]


async def _backward_unary(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, timeout: float, **kwargs
) -> List[torch.Tensor]:
    grad_inputs: runtime_pb2.ExpertResponse = await stub.rpc_backward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors), **kwargs),
        timeout=timeout,
    )
    return [deserialize_torch_tensor(t) for t in grad_inputs.tensors]


async def _forward_stream(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, timeout: float, **kwargs
) -> List[torch.Tensor]:
    parts = (
        runtime_pb2.ExpertRequest(uid=uid, tensors=[part], **kwargs)
        for tensor in serialized_tensors
        for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
    )
    outputs = await asyncio.wait_for(stub.rpc_forward_stream(iter_as_aiter(parts)), timeout)
    outputs = aiter_with_timeout(outputs, timeout)
    return await deserialize_tensor_stream(msg.tensors async for msg in outputs)


async def _backward_stream(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, timeout: float, **kwargs
) -> List[torch.Tensor]:
    parts = (
        runtime_pb2.ExpertRequest(uid=uid, tensors=[part], **kwargs)
        for tensor in serialized_tensors
        for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
    )
    grad_inputs = await asyncio.wait_for(stub.rpc_backward_stream(iter_as_aiter(parts)), timeout)
    grad_inputs = aiter_with_timeout(grad_inputs, timeout)
    return await deserialize_tensor_stream(msg.tensors async for msg in grad_inputs)


async def run_remote_forward(
    uid: ModuleUID,
    stub: StubBase,
    rpc_info: RPCInfo,
    *inputs: torch.Tensor,
    timeout: float,
    metadata: Optional[bytes] = None,
    **kwargs,
) -> Tuple[torch.Tensor, ...]:
    """
    Serializes input tensors and calls "rpc_forward" on a remote server.
    Mostly adapted from https://github.com/learning-at-home/hivemind/blob/7a7c93aefffc9494c39e7b170c07cb06d8c09c4c/hivemind/moe/client/expert.py#L198
    but without RemoteExpertWorker.run_coroutine() call that leads to deadlock here.
    """

    # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
    # detach to avoid pickling the computation graph
    assert len(kwargs) == len(rpc_info["keyword_names"]), f"Keyword args should be {rpc_info['keyword_names']}"
    kwargs = {key: kwargs[key] for key in rpc_info["keyword_names"]}

    # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors
    forward_inputs = (inputs, kwargs)

    # Modify forward_schema to support prompts
    args_schema, kwargs_schema = rpc_info["forward_schema"]
    # TODO: rm this assert when support arbitrary number of input tensors
    assert len(args_schema) == 1 and len(inputs) == 2
    forward_schema_with_prompts = (tuple(args_schema * len(inputs)), kwargs_schema)

    if not nested_compare(forward_inputs, forward_schema_with_prompts):
        raise TypeError(f"Inputs do not match expert input schema. Did you pass the right number of parameters?")

    forward_inputs = nested_flatten(forward_inputs)
    inputs = tuple(tensor.cpu().detach() for tensor in forward_inputs)

    # Asynchronous serialization
    loop = asyncio.get_running_loop()
    serialized_tensors = await asyncio.gather(
        *(
            loop.run_in_executor(None, serialize_torch_tensor, tensor.to(proto.dtype), proto.compression)
            for tensor, proto in zip(inputs, nested_flatten(forward_schema_with_prompts))
        )
    )

    # call RPC on remote server
    size = sum(t.element_size() * t.nelement() for t in inputs)
    forward_fn = _forward_stream if size > MAX_UNARY_PAYLOAD_SIZE else _forward_unary
    deserialized_outputs = await forward_fn(uid, serialized_tensors, stub, timeout, metadata=metadata, **kwargs)
    return nested_pack(deserialized_outputs, structure=rpc_info["outputs_schema"])


async def run_remote_backward(
    uid: ModuleUID,
    stub: StubBase,
    rpc_info: RPCInfo,
    inputs: torch.Tensor,
    grad_outputs: List[torch.Tensor],
    *extra_tensors: torch.Tensor,
    timeout: float,
    metadata: Optional[bytes] = None,
    **kwargs,
) -> Sequence[torch.Tensor]:
    """
    Serializes grad outputs and calls "rpc_backward" on a remote server.
    Mostly adapted from https://github.com/learning-at-home/hivemind/blob/7a7c93aefffc9494c39e7b170c07cb06d8c09c4c/hivemind/moe/client/expert.py#L221
    but without RemoteExpertWorker.run_coroutine() call that leads to deadlock here.
    """

    grad_outputs_cpu = tuple(tensor.cpu() for tensor in grad_outputs)
    inputs_and_grad_outputs = tuple(nested_flatten((inputs, grad_outputs_cpu, *extra_tensors)))

    # Modify forward_schema to support prompts
    args_schema, kwargs_schema = rpc_info["forward_schema"]
    assert len(args_schema) == 1 and isinstance(inputs, torch.Tensor)
    # TODO generalize this
    prompts_schema = next(iter(args_schema))
    backward_schema = tuple(nested_flatten((rpc_info["forward_schema"], rpc_info["outputs_schema"], prompts_schema)))

    # Asynchronous serialization
    loop = asyncio.get_running_loop()
    serialized_tensors = await asyncio.gather(
        *(
            loop.run_in_executor(None, serialize_torch_tensor, tensor.to(proto.dtype), proto.compression)
            for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)
        )
    )

    size = sum(t.element_size() * t.nelement() for t in inputs_and_grad_outputs)
    backward_fn = _backward_stream if size > MAX_UNARY_PAYLOAD_SIZE else _backward_unary
    deserialized_grad_inputs = await backward_fn(uid, serialized_tensors, stub, timeout, metadata=metadata, **kwargs)
    return deserialized_grad_inputs
