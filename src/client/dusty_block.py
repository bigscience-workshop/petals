from __future__ import annotations

import inspect
from functools import wraps
from typing import AsyncIterator, Callable, Optional

from hivemind.moe.client import RemoteExpert
from hivemind.moe.expert_uid import ExpertInfo
from hivemind.p2p import P2P, StubBase
from hivemind.proto import runtime_pb2
from hivemind.utils import MSGPackSerializer, amap_in_executor

from src.client.dust_bank import DustBankBase


class DustyRemoteBlock(RemoteExpert):
    def __init__(self, bank: DustBankBase, expert_info: ExpertInfo, p2p: P2P):
        self._bank = bank
        super().__init__(expert_info, p2p)

    def _unary_request_wrapper(self, rpc_call: Callable, rpc_name: str):
        @wraps(rpc_call)
        async def rpc(input: runtime_pb2.ExpertRequest, timeout: Optional[float] = None):
            meta = MSGPackSerializer.loads(input.metadata) if input.metadata else {}
            meta["__dust"] = self._bank.get_dust(input, rpc_name)
            input.metadata = MSGPackSerializer.dumps(meta)
            return await rpc_call(input, timeout)

        return rpc

    def _stream_request_wrapper(self, rpc_call: Callable, rpc_name: str):
        @wraps(rpc_call)
        async def rpc(input: AsyncIterator[runtime_pb2.ExpertRequest], timeout: Optional[float] = None):
            is_meta_set = False

            def _metadata_setter(chunk: runtime_pb2.ExpertRequest) -> runtime_pb2.ExpertRequest:
                nonlocal is_meta_set
                if not is_meta_set:
                    meta = MSGPackSerializer.loads(chunk.metadata) if chunk.metadata else {}
                    meta["__dust"] = self._bank.get_dust(chunk, rpc_name)
                    chunk.metadata = MSGPackSerializer.dumps(meta)
                    is_meta_set = True
                return chunk

            return await rpc_call(amap_in_executor(_metadata_setter, input), timeout)

        return rpc

    def _dustify_handler_stub(self, stub: StubBase) -> StubBase:
        for name, method in inspect.getmembers(stub, predicate=inspect.ismethod):
            if name.startswith("rpc"):
                spec = inspect.getfullargspec(method)
                # rpc callers has 3 arguments: stub, input and timeout
                if len(spec.args) != 3:
                    continue

                input_type = spec.annotations[spec.args[1]]

                if input_type is AsyncIterator[runtime_pb2.ExpertRequest]:
                    setattr(stub, name, self._stream_request_wrapper(method, name))
                elif input_type is runtime_pb2.ExpertRequest:
                    setattr(stub, name, self._unary_request_wrapper(method, name))
        return stub

    @property
    def _stub(self) -> StubBase:
        return super().stub

    @property
    def stub(self) -> StubBase:
        return self._dustify_handler_stub(self._stub)
