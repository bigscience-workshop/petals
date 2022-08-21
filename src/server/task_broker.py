from abc import ABC, abstractmethod

from hivemind.moe.server.task_pool import Task


class DustBrokerBase(ABC):
    """Abstract class for DustBroker whose reponsibility is to evaluate task profit"""

    @abstractmethod
    def __call__(self, task: Task, dust: float) -> float:
        """Evaluates task value by the amout of dust promised"""
        pass


class SimpleBroker(DustBrokerBase):
    """Simple implementation of DustBroker which counts amount of dust per task size"""

    def __call__(self, task: Task, dust: float) -> float:
        # TODO: this was taken from original task pool. Is is right?
        task_size = len(task.args[0]) if task.args else 1
        return dust / task_size
