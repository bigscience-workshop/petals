from abc import ABC, abstractmethod

import torch
from hivemind.moe.server.task_pool import Task


class TaskPrioritizerBase(ABC):
    """Abstract class for DustBroker whose reponsibility is to evaluate task profit"""

    @abstractmethod
    def prioritize(self, *input: torch.Tensor, points: float = 0.0, **kwargs) -> float:
        """Evaluates task value by the amout of points given"""
        pass


class DummyTaskPrioritizer(TaskPrioritizerBase):
    """Simple implementation of DustBroker which counts amount of dust per task size"""

    def __call__(self, *input: torch.Tensor, points: float = 0.0, **kwargs) -> float:
        return 0.0
