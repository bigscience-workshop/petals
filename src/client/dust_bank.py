import inspect
from abc import ABC, abstractmethod
from functools import wraps
from typing import AsyncIterator, Callable, Optional

from hivemind.p2p import StubBase
from hivemind.proto import runtime_pb2
from hivemind.proto.runtime_pb2 import ExpertRequest
from hivemind.utils import MSGPackSerializer, amap_in_executor


class DustBankBase(ABC):
    @abstractmethod
    def get_dust(self, request: ExpertRequest, method_name: str) -> float:
        pass


class DummyDustBank(DustBankBase):
    def get_dust(self, request: ExpertRequest, method_name: str) -> float:
        return 0.0


def _unary_request_wrapper(rpc_call: Callable, rpc_name: str, bank: DustBankBase):
    @wraps(rpc_call)
    async def rpc(stub: StubBase, input: runtime_pb2.ExpertRequest, timeout: Optional[float] = None):
        meta = MSGPackSerializer.loads(input.metadata) if input.metadata else {}
        meta.update("__dust", bank.get_dust(input, rpc_name))
        input.metadata = MSGPackSerializer.dumps(meta)
        return await rpc_call(stub, input, timeout)

    return rpc


def _stream_request_wrapper(rpc_call: Callable, rpc_name: str, bank: DustBankBase):
    @wraps(rpc_call)
    async def rpc(stub: StubBase, input: AsyncIterator[runtime_pb2.ExpertRequest], timeout: Optional[float] = None):
        is_meta_set = False

        def _metadata_setter(chunk: runtime_pb2.ExpertRequest) -> runtime_pb2.ExpertRequest:
            nonlocal is_meta_set
            if not is_meta_set:
                meta = MSGPackSerializer.loads(chunk.metadata) if chunk.metadata else {}
                meta.update("__dust", bank.get_dust(chunk, rpc_name))
                chunk.metadata = MSGPackSerializer.dumps(meta)
                is_meta_set = True
            return chunk

        return await rpc_call(stub, amap_in_executor(_metadata_setter, input), timeout)

    return rpc


def _dustify_handler_stub(stub: StubBase, bank: DustBankBase) -> StubBase:
    for name, method in inspect.getmembers(stub, predicate=inspect.ismethod):
        if name.startswith("rpc"):
            spec = inspect.getfullargspec(method)
            # rpc callers has 3 arguments: stub, input and timeout
            if len(spec.args) != 3:
                continue

            input_type = spec.annotations[spec.args[1]]

            if input_type is AsyncIterator[runtime_pb2.ExpertRequest]:
                setattr(stub, name, _stream_request_wrapper(method, name, bank))
            elif input_type is runtime_pb2.ExpertRequest:
                setattr(stub, name, _unary_request_wrapper(method, name, bank))
    return stub


def payment_wrapper(bank: DustBankBase) -> Callable:
    def class_wrapper(cls):
        d = cls.__dict__
        if "stub" not in d or not isinstance(d["stub"], property):
            raise TypeError('wrapped module class supposed to have property "stub"')
        old_stub = d["stub"]

        def _stub(self):
            stub = old_stub.__get__(self)
            return _dustify_handler_stub(stub, bank)

        return type(cls.__name__, cls.__bases__, {k: v if k != "stub" else property(_stub) for k, v in d.items()})

    return class_wrapper
