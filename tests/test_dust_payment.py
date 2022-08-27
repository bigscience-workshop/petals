from typing import AsyncIterator, Optional

import pytest
import torch
from hivemind.compression import deserialize_tensor_stream, deserialize_torch_tensor, serialize_torch_tensor
from hivemind.proto.runtime_pb2 import ExpertRequest
from hivemind.utils import MSGPackSerializer, amap_in_executor, iter_as_aiter, split_for_streaming

from src.client.dust_bank import DustBankBase
from src.client.dusty_block import DustyRemoteBlock


class DustBankTest(DustBankBase):
    def __init__(self):
        self._p = {
            "rpc_single": 1,
            "rpc_stream": 2,
        }

    def get_dust(self, request: ExpertRequest, method_name: str) -> float:
        return self._p.get(method_name, -1)


class HandlerStubTest:
    async def rpc_single(self, input: ExpertRequest, timeout: Optional[float] = None):
        return input

    async def rpc_stream(self, input: AsyncIterator[ExpertRequest], timeout: Optional[float] = None):
        return input

    async def rpc_info(self, input: str, timeout: Optional[float] = None):
        return input


class RemoteBlockTest(DustyRemoteBlock):
    @property
    def _stub(self):
        return HandlerStubTest()


@pytest.mark.asyncio
async def test_single():
    remote = RemoteBlockTest(DustBankTest(), None, None)
    stub = remote.stub
    input = torch.randn(1, 2)
    request = ExpertRequest(uid="expert1", tensors=[serialize_torch_tensor(input)])

    print(stub)
    out: ExpertRequest = await stub.rpc_single(request)

    assert out.metadata != b""
    assert len(out.tensors) == 1
    assert torch.allclose(input, deserialize_torch_tensor(out.tensors[0]))

    meta = MSGPackSerializer.loads(out.metadata)
    assert isinstance(meta, dict)
    assert "__dust" in meta
    assert meta["__dust"] == 1


@pytest.mark.asyncio
async def test_stream():
    remote = RemoteBlockTest(DustBankTest(), None, None)
    stub = remote.stub
    input = torch.randn(2**21, 2)

    split = (p for t in [serialize_torch_tensor(input)] for p in split_for_streaming(t, chunk_size_bytes=2**16))
    output_generator = await stub.rpc_stream(
        amap_in_executor(
            lambda tensor_part: ExpertRequest(uid="expert2", tensors=[tensor_part]),
            iter_as_aiter(split),
        ),
    )
    outputs_list = [part async for part in output_generator]
    assert len(outputs_list) == 2**5 * 8
    assert outputs_list[0].metadata != b""
    for i in range(1, len(outputs_list)):
        assert outputs_list[i].metadata == b""

    meta = MSGPackSerializer.loads(outputs_list[0].metadata)
    assert isinstance(meta, dict)
    assert "__dust" in meta
    assert meta["__dust"] == 2

    results = await deserialize_tensor_stream(amap_in_executor(lambda r: r.tensors, iter_as_aiter(outputs_list)))
    assert len(results) == 1
    assert torch.allclose(results[0], input)


@pytest.mark.asyncio
async def test_no_wrapper():
    remote = RemoteBlockTest(DustBankTest(), None, None)
    stub = remote.stub

    test = await stub.rpc_info("Test")
    assert test == "Test"
