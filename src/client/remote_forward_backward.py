"""
Utility functions that call RPC forward or backward on a single remote server
"""
from typing import Iterable, List, Sequence

import torch
from hivemind.compression.serialization import deserialize_tensor_stream, deserialize_torch_tensor
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, MAX_UNARY_PAYLOAD_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import amap_in_executor, iter_as_aiter
from hivemind.utils.streaming import split_for_streaming


async def _backward_stream(uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub) -> List[torch.Tensor]:
    split = (part for tensor in serialized_tensors for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE))

    grad_inputs = await stub.rpc_backward_stream(
        amap_in_executor(
            lambda tensor: runtime_pb2.ExpertRequest(uid=uid, tensors=[tensor]),
            iter_as_aiter(split),
        ),
    )
    tensors_stream = amap_in_executor(lambda msg: msg.tensors, grad_inputs)
    return await deserialize_tensor_stream(tensors_stream)


async def _backward_unary(uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub) -> List[torch.Tensor]:
    grad_inputs: runtime_pb2.ExpertResponse = await stub.rpc_backward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors))
    )
    return [deserialize_torch_tensor(t) for t in grad_inputs.tensors]


async def remote_backward(
    uid: str, inputs_and_grads: Sequence[torch.Tensor], serialized_tensors: Iterable[runtime_pb2.Tensor], stub
) -> List[torch.Tensor]:
    """Call rpc_backward (unary or stream) on a single remote server, return grads w.r.t. arguments"""
    size = 0
    for t in inputs_and_grads:
        size += t.element_size() * t.nelement()
        if size > MAX_UNARY_PAYLOAD_SIZE:
            return await _backward_stream(uid, serialized_tensors, stub)
    else:
        return await _backward_unary(uid, serialized_tensors, stub)


async def _forward_stream(uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub) -> List[torch.Tensor]:
    split = (p for t in serialized_tensors for p in split_for_streaming(t, DEFAULT_MAX_MSG_SIZE))

    outputs = await stub.rpc_forward_stream(
        amap_in_executor(
            lambda tensor: runtime_pb2.ExpertRequest(uid=uid, tensors=[tensor]),
            iter_as_aiter(split),
        ),
    )

    tensors_stream = amap_in_executor(lambda msg: msg.tensors, outputs)
    return await deserialize_tensor_stream(tensors_stream)


async def _forward_unary(uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub) -> List[torch.Tensor]:
    outputs: runtime_pb2.ExpertResponse = await stub.rpc_forward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors))
    )
    return [deserialize_torch_tensor(t) for t in outputs.tensors]


async def remote_forward(
    uid: str, inputs: Sequence[torch.Tensor], serialized_tensors: Iterable[runtime_pb2.Tensor], stub
) -> List[torch.Tensor]:
    """Call rpc_forward (unary or stream) on a single remote server, return block outputs"""
    size = 0
    for t in inputs:
        size += t.element_size() * t.nelement()
        if size > MAX_UNARY_PAYLOAD_SIZE:
            return await _forward_stream(uid, serialized_tensors, stub)
    else:
        return await _forward_unary(uid, serialized_tensors, stub)
