from abc import ABC, abstractmethod

from hivemind.moe.server.task_pool import Task


class TaskPrioritizerBase(ABC):
    """Abstract class for DustBroker whose reponsibility is to evaluate task profit"""

    @abstractmethod
    def prioritize(self, task: Task, points: float, *args, **kwargs) -> float:
        """Evaluates task value by the amout of points given"""
        pass


class DummyTaskPrioritizer(TaskPrioritizerBase):
    """Simple implementation of DustBroker which counts amount of dust per task size"""

    def __call__(self, task: Task, points: float, *args, **kwargs) -> float:
        return 0.0
