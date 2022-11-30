from abc import ABC, abstractmethod

import torch
from hivemind.moe.server.task_pool import Task


class TaskPrioritizerBase(ABC):
    """Abstract class for TaskPrioritizer whose reponsibility is to evaluate task priority"""

    @abstractmethod
    def prioritize(self, *input: torch.Tensor, points: float = 0.0, **kwargs) -> float:
        """Evaluates task value by the amout of points given, task input and additional kwargs. Lower priority is better"""
        pass


class DummyTaskPrioritizer(TaskPrioritizerBase):
    """Simple implementation of TaskPrioritizer which gives constant zero priority for every task"""

    def prioritize(self, *input: torch.Tensor, points: float = 0.0, **kwargs) -> float:
        return 0.0
