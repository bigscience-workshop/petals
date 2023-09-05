"""
Utility functions that call RPC forward or backward on a single remote server
"""
import asyncio
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from hivemind import nested_compare, nested_flatten, nested_pack, serialize_torch_tensor, PeerID
from hivemind.compression.serialization import deserialize_tensor_stream, deserialize_torch_tensor
from hivemind.p2p import StubBase
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, MAX_UNARY_PAYLOAD_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import aiter_with_timeout, iter_as_aiter
from hivemind.utils.streaming import split_for_streaming
from hivemind.utils.tensor_descr import BatchTensorDescriptor

from petals import RemoteSequenceManager
from petals.client.config import ClientConfig
from petals.data_structures import ModuleUID, RPCInfo, CHAIN_DELIMITER
from petals.server.handler import TransformerConnectionHandler
from petals.utils.packaging import pack_args_kwargs


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
    sequence_manager: RemoteSequenceManager,
    peer_id: PeerID,
    span_uids: Sequence[ModuleUID],
    *args: torch.Tensor,
    **kwargs: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """
    Serializes input tensors and calls "rpc_forward" on a remote server.
    Mostly adapted from https://github.com/learning-at-home/hivemind/blob/7a7c93aefffc9494c39e7b170c07cb06d8c09c4c/hivemind/moe/client/expert.py#L198
    but without RemoteExpertWorker.run_coroutine() call that leads to deadlock here.
    """
    merged_uid = CHAIN_DELIMITER.join(span_uids)
    stub = TransformerConnectionHandler.get_stub(sequence_manager.state.p2p, peer_id)
    flat_inputs, args_structure = pack_args_kwargs(*args, **kwargs)
    metadata = sequence_manager.get_request_metadata(peer_id, "rpc_forward", span_uids, *args, **kwargs)
    compressions = sequence_manager.get_compression_codecs(peer_id, "rpc_forward", span_uids, *args, **kwargs)
    if compressions is None:
        compressions = [runtime_pb2.CompressionType.NONE] * len(flat_inputs)
    compressions = list(nested_flatten(compressions))
    assert len(compressions) == len(flat_inputs), f"got {len(flat_inputs)} tensors but {len(compressions)} codecs"
    inputs = tuple(tensor.cpu().detach().requires_grad_(tensor.requires_grad) for tensor in flat_inputs)

    # Asynchronous serialization
    loop = asyncio.get_running_loop()
    serialized_tensors = await asyncio.gather(
        *(
            loop.run_in_executor(None, serialize_torch_tensor, tensor, compression)
            for tensor, compression in zip(inputs, compressions)
        )
    )

    # call RPC on remote server
    size = sum(t.element_size() * t.nelement() for t in inputs)
    forward_fn = _forward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else _forward_unary
    # Hotfix: we use "// 2" since hivemind==1.1.5 serializes bfloat16 tensors in float32, so they take 2x more space - TODO remove in the next PR
    return await forward_fn(merged_uid, serialized_tensors, stub, sequence_manager.config, metadata=metadata)


async def run_remote_backward(
    sequence_manager: RemoteSequenceManager,
    span_uids: Sequence[ModuleUID],
    stub: StubBase,
    grad_outputs: Sequence[torch.Tensor],
    *args: torch.Tensor,
    **kwargs: torch.Tensor,
) -> Sequence[torch.Tensor]:
    """
    Serializes grad outputs and calls "rpc_backward" on a remote server.
    Mostly adapted from https://github.com/learning-at-home/hivemind/blob/7a7c93aefffc9494c39e7b170c07cb06d8c09c4c/hivemind/moe/client/expert.py#L221
    but without RemoteExpertWorker.run_coroutine() call that leads to deadlock here.
    """
    flat_tensors, args_structure = pack_args_kwargs(
        [grad.cpu() for grad in grad_outputs], args, kwargs
    )
    metadata = sequence_manager.get_request_metadata(
        "rpc_backward", args_structure, span_uids, *flat_tensors, peer_id=span.peer_id
    )

    # Asynchronous serialization
    loop = asyncio.get_running_loop()
    serialized_tensors = await asyncio.gather(
        *(
            loop.run_in_executor(None, serialize_torch_tensor, compression)
            for tensor, proto in zip(flat_inputs_and_grad_outputs, backward_schema)
        )
    )

    size = sum(t.element_size() * t.nelement() for t in flat_inputs_and_grad_outputs)
    backward_fn = _backward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else _backward_unary
    # Hotfix: we use "// 2" since hivemind==1.1.5 serializes bfloat16 tensors in float32, so they take 2x more space
    return await backward_fn(uid, serialized_tensors, stub, config, metadata=metadata)
