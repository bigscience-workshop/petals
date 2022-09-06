import multiprocessing as mp
import os
import threading
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from queue import PriorityQueue, Empty
from typing import Sequence

import torch
from hivemind import MPFuture, use_hivemind_log_handler, get_logger
from hivemind.moe.server.task_pool import TaskPoolBase


use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


@dataclass(order=True, frozen=True)
class PrioritizedTask:
    priority: float
    future: MPFuture = field(compare=False)
    args: Sequence[torch.Tensor]  = field(compare=False)


class PrioritizedTaskPool(TaskPoolBase):
    """

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.min_batch_size == 1, "PriorityTaskPool supports no batching"

        self.priority_queue = mp.Queue(maxsize=self.tasks._maxsize)
        self.prioritized_task_queue = PriorityQueue(maxsize=self.tasks._maxsize)

    def submit_task(self, *args: torch.Tensor, priority: float = 0.0) -> Future:
        f = super().submit_task(*args)
        self.priority_queue.put(priority)
        # TODO use a single queue here
        return f

    def _priortize_tasks(self):
        """Infinite loop prioritizing incoming tasks"""
        while True:
            task = self.tasks.get(block=True)
            priority = self.priority_queue.get(block=True)
            self.prioritized_task_queue.put(PrioritizedTask(priority, task), block=True)

    def run(self, *args, **kwargs):
        torch.set_num_threads(1)
        logger.info(f"{self.name} starting, pid={os.getpid()}")
        pending_batches = {}  # Dict[batch uuid, List[MPFuture]] for each batch currently in runtime

        output_thread = threading.Thread(
            target=self._pool_output_loop, args=[pending_batches], name=f"{self.name}_output", daemon=True
        )
        priority_thread = threading.Thread(
            target=self._priortize_tasks, args=[], name=f"{self.name}_priority", daemon=True
        )

        try:
            output_thread.start()
            priority_thread.start()
            self._pool_input_loop(pending_batches, *args, **kwargs)
        except KeyboardInterrupt:
            logger.debug("Caught KeyboardInterrupt, shutting down")
        finally:
            output_thread.join()
            priority_thread.join()

    # TODO: this is a copy-paste of the original method, except that we use different queue
    def iterate_minibatches(self, *args, **kwargs):
        """Form minibatches by grouping one or more tasks together up to self.max_batch_size"""
        while True:
            try:
                logger.debug(f"{self.name} getting next task")
                task: PrioritizedTask = self.prioritized_task_queue.get(timeout=self.timeout)
            except Empty:
                logger.warning(f"Timeout reached but batch doesn't contain >={self.min_batch_size} elements yet")
                continue

            try:
                if task.task.future.set_running_or_notify_cancel():
                    yield [task.task]
            except InvalidStateError as e:
                logger.debug(f"Failed to add task to batch: {task.task.future} raised {e}")
