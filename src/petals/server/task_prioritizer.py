from abc import ABC, abstractmethod

import torch


class TaskPrioritizerBase(ABC):
    """Abstract class for TaskPrioritizer whose responsibility is to evaluate task priority"""

    @abstractmethod
    def prioritize(self, *input: torch.Tensor, points: float = 0.0, **kwargs) -> float:
        """Evaluates task value by the amount of points given, task input and additional kwargs. Lower priority is better"""
        pass


class DummyTaskPrioritizer(TaskPrioritizerBase):
    """Simple implementation of TaskPrioritizer which gives constant zero priority for every task"""

    def prioritize(self, *input: torch.Tensor, points: float = 0.0, **kwargs) -> float:
        if kwargs.get("type") == "inference":
            return 1.0  # inference steps go first since they are more latency-sensitive
        return 2.0  # forward, backward
