from abc import ABC, abstractmethod

from hivemind.moe.server.task_pool import Task


class TaskBrokerBase(ABC):
    @abstractmethod
    def __call__(self, task: Task, pollen: float) -> float:
        pass


class SimpleBroker(TaskBrokerBase):
    def __call__(self, task: Task, pollen: float) -> float:
        task_size = len(task.args[0]) if task.args else 1
        return pollen / task_size
