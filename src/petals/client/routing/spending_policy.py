"""
An interface for exchanging internal "BLOOM points" for higher priority compute requests. NOT IMPLEMENTED.
The intent is to let Petals participants earn points by helping others while idle (e.g. at night), then use these
 points to run their own compute experiments faster. See Section 4 of https://arxiv.org/abs/2209.01188 for discussion.
"""
from abc import ABC, abstractmethod


class SpendingPolicyBase(ABC):
    @abstractmethod
    def get_points(self, protocol: str, *args, **kwargs) -> float:
        pass


class NoSpendingPolicy(SpendingPolicyBase):
    def get_points(self, protocol: str, *args, **kwargs) -> float:
        return 0.0
