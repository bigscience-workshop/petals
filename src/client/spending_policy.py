from abc import ABC, abstractmethod

from hivemind.proto.runtime_pb2 import ExpertRequest


class SpendingPolicyBase(ABC):
    @abstractmethod
    def get_points(self, request: ExpertRequest, method_name: str, *args, **kwargs) -> float:
        pass


class DummySpendingPolicy(SpendingPolicyBase):
    def get_points(self, request: ExpertRequest, method_name: str, *args, **kwargs) -> float:
        return 0.0
