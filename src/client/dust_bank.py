from abc import ABC, abstractmethod
from functools import wraps

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
