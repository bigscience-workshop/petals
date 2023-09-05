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
from hivemind.utils.tensor_descr import BatchTensorDescriptor

from petals.client.config import ClientConfig
from petals.data_structures import ModuleUID, RPCInfo


async def _forward_unary(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, config: ClientConfig
) -> List[torch.Tensor]:
    outputs: runtime_pb2.ExpertResponse = await stub.rpc_forward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors)),
        timeout=config.request_timeout,
    )
    return [deserialize_torch_tensor(t) for t in outputs.tensors]


async def _backward_unary(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, config: ClientConfig
) -> List[torch.Tensor]:
    grad_inputs: runtime_pb2.ExpertResponse = await stub.rpc_backward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors)),
        timeout=config.request_timeout,
    )
    return [deserialize_torch_tensor(t) for t in grad_inputs.tensors]


async def _forward_stream(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, config: ClientConfig
) -> List[torch.Tensor]:
    parts = (
        runtime_pb2.ExpertRequest(uid=uid, tensors=[part])
        for tensor in serialized_tensors
        for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
    )
    outputs = await asyncio.wait_for(stub.rpc_forward_stream(iter_as_aiter(parts)), config.connect_timeout)
    outputs = aiter_with_timeout(outputs, config.request_timeout)
    return await deserialize_tensor_stream(msg.tensors async for msg in outputs)


async def _backward_stream(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, config: ClientConfig
) -> List[torch.Tensor]:
    parts = (
        runtime_pb2.ExpertRequest(uid=uid, tensors=[part])
        for tensor in serialized_tensors
        for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
    )
    grad_inputs = await asyncio.wait_for(stub.rpc_backward_stream(iter_as_aiter(parts)), config.connect_timeout)
    grad_inputs = aiter_with_timeout(grad_inputs, config.request_timeout)
    return await deserialize_tensor_stream(msg.tensors async for msg in grad_inputs)


async def run_remote_forward(
    uid: ModuleUID,
    stub: StubBase,
    rpc_info: RPCInfo,
    *forward_inputs: torch.Tensor,
    config: ClientConfig,
    metadata: Optional[bytes] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Serializes input tensors and calls "rpc_forward" on a remote server.
    Mostly adapted from https://github.com/learning-at-home/hivemind/blob/7a7c93aefffc9494c39e7b170c07cb06d8c09c4c/hivemind/moe/client/expert.py#L198
    but without RemoteExpertWorker.run_coroutine() call that leads to deadlock here.
    """
    args_schema, kwargs_schema = rpc_info["forward_schema"]
    compression = args_schema[0].compression
    forward_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in forward_inputs)
    inputs = tuple(tensor.cpu().detach() for tensor in forward_inputs)
    # Asynchronous serialization
    loop = asyncio.get_running_loop()
    serialized_tensors = await asyncio.gather(
        *(
            loop.run_in_executor(None, serialize_torch_tensor, tensor.to(proto.dtype), proto.compression)
            for tensor, proto in zip(inputs, forward_schema)
        )
    )

    # call RPC on remote server
    size = sum(t.element_size() * t.nelement() for t in inputs)
    forward_fn = _forward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else _forward_unary
    # Hotfix: we use "// 2" since hivemind==1.1.5 serializes bfloat16 tensors in float32, so they take 2x more space
    deserialized_outputs = await forward_fn(uid, serialized_tensors, stub, config, metadata=metadata)
    return nested_pack(deserialized_outputs, structure=rpc_info["outputs_schema"])


async def run_remote_backward(
    uid: ModuleUID,
    stub: StubBase,
    rpc_info: RPCInfo,
    *inputs_and_grad_outputs: torch.Tensor,
    config: ClientConfig,
    metadata: Optional[bytes] = None,
    **kwargs,
) -> Sequence[torch.Tensor]:
    """
    Serializes grad outputs and calls "rpc_backward" on a remote server.
    Mostly adapted from https://github.com/learning-at-home/hivemind/blob/7a7c93aefffc9494c39e7b170c07cb06d8c09c4c/hivemind/moe/client/expert.py#L221
    but without RemoteExpertWorker.run_coroutine() call that leads to deadlock here.
    """
    args_schema, kwargs_schema = rpc_info["forward_schema"]
    outputs_schema = rpc_info["outputs_schema"]
    compression = args_schema[0].compression
    backward_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in inputs_and_grad_outputs)
    # TODO: create more explicit way to check servers schema and client's structure
    assert (
        len(inputs_and_grad_outputs) >= len(args_schema) + len(outputs_schema) + 1
    ), "Inputs, grad_outputs and prompt tensors are necessary for a backward step"

    # Asynchronous serialization
    loop = asyncio.get_running_loop()
    serialized_tensors = await asyncio.gather(
        *(
            loop.run_in_executor(None, serialize_torch_tensor, tensor.to(proto.dtype), proto.compression)
            for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)
        )
    )

    size = sum(t.element_size() * t.nelement() for t in inputs_and_grad_outputs)
    backward_fn = _backward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else _backward_unary
    # Hotfix: we use "// 2" since hivemind==1.1.5 serializes bfloat16 tensors in float32, so they take 2x more space
    deserialized_grad_inputs = await backward_fn(uid, serialized_tensors, stub, config, metadata=metadata, **kwargs)
    return deserialized_grad_inputs
