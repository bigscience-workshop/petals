"""Code for serving bloom blocks via hivemind-server"""
import ctypes
import multiprocessing as mp
import os
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue
from typing import Optional, Sequence, Tuple, Dict, Any, List

import torch
from hivemind import use_hivemind_log_handler
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.moe.server.task_pool import Task, TaskPool
from hivemind.utils import InvalidStateError, get_logger

from src.bloom.from_pretrained import BloomBlock
from src.server.cache import MemoryCache

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


@dataclass(order=True)
class PrioritizedTask:
    priority: float
    task: Task = field(compare=False)


class PrioritizedTaskPool(TaskPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.min_batch_size == 1, "PriorityTaskPool supports no batching"

        self.priority_queue = mp.Queue(maxsize=self.tasks._maxsize)
        self.prioritized_task_queue = PriorityQueue(maxsize=self.tasks._maxsize)
        self.undispatched_task_priorities = mp.SimpleQueue()
        self._timestamp = mp.Value(ctypes.c_double, 1.0)

    @property
    def priority(self):
        return (-self._priority.value, -self._timestamp.value)

    @priority.setter
    def priority(self, priority_tuple: Sequence[float]):
        assert len(priority_tuple) == 2, "pool priority must be a tuple of (priority, time_submitted)"
        self._priority.value, self._timestamp.value = map(float, priority_tuple)

    def submit_task(self, *args: torch.Tensor, priority: float = 0.0) -> Future:
        f = super().submit_task(*args)
        self.priority_queue.put(priority)
        self.undispatched_task_priorities.put(priority)
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

    def _pool_input_loop(self, pending_batches: Dict[Any, List[Task]], *args, **kwargs):
        """Infinite loop: aggregate tasks into batches and send them to runtime"""

        prev_num_tasks = 0  # number of tasks currently in shared buffer
        batch_index = max(pending_batches.keys(), default=0)
        batch_iterator = self.iterate_minibatches(*args, **kwargs)

        while True:
            # SIDE-EFFECT - compute pool priority from timestamp of earliest undispatched task
            # assumes that tasks are processed in the same order as they are created
            for skip_i in range(prev_num_tasks):
                dispatched_task_timestamp = self.undispatched_task_timestamps.get()
                dispatched_task_priority = self.undispatched_task_priorities.get()
                if skip_i == prev_num_tasks - 1:
                    self.priority = (dispatched_task_priority, dispatched_task_timestamp)

            logger.debug(f"{self.name} getting next batch")
            batch_tasks = next(batch_iterator)
            # save batch futures, _output_loop will deliver on them later
            pending_batches[batch_index] = batch_tasks

            logger.debug(f"{self.name}, batch  {batch_index}: aggregating inputs")
            # find or create shared arrays for current batch size
            batch_inputs = [torch.cat([task.args[i] for task in batch_tasks]) for i in range(len(batch_tasks[0].args))]
            batch_inputs = [inp.detach().requires_grad_(inp.requires_grad).share_memory_() for inp in batch_inputs]

            logger.debug(f"{self.name}, batch {batch_index}: sending to runtime")
            self.batch_sender.send((batch_index, batch_inputs))
            logger.debug(f"{self.name}, batch {batch_index}: sent to runtime")
            prev_num_tasks = len(batch_tasks)
            batch_index += 1


    # TODO: this is a copy-paste of the original method, except that we use different queue
    def iterate_minibatches(self, *args, **kwargs):
        """Form minibatches by grouping one or more tasks together up to self.max_batch_size"""
        print('IN iterate_minibatches')
        while True:
            try:
                logger.debug(f"{self.name} getting next task")
                task: PrioritizedTask = self.prioritized_task_queue.get(timeout=self.timeout)
                print('IN iterate_minibatches - 1')
            except Empty:
                logger.warning(f"Timeout reached but batch doesn't contain >={self.min_batch_size} elements yet")
                print('IN iterate_minibatches - 2')
                continue

            print('IN iterate_minibatches - 3')
            try:
                if task.task.future.set_running_or_notify_cancel():
                    print('IN iterate_minibatches - 4')
                    yield [task.task]
                    print('IN iterate_minibatches - 5')
            except InvalidStateError as e:
                logger.debug(f"Failed to add task to batch: {task.task.future} raised {e}")


class TransformerBackend(ModuleBackend):
    """A wrapper for BloomBlock that can process requests for bloom layer forward, forward_incremental, and backward"""

    def __init__(self, *args, memory_cache: MemoryCache, backend_dtype: Optional[torch.dtype] = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.module, BloomBlock)
        self.memory_cache = memory_cache
        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Bloom layer parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Bloom layer parameters must not accumulate gradients, but {name} does"

        self.inference_pool = PrioritizedTaskPool(
            self.inference_step, max_batch_size=self.forward_pool.max_batch_size, name=f"{self.name}_inference"
        )
        self.forward_pool = PrioritizedTaskPool(self.forward, name=f"{self.name}_forward", **kwargs)
        self.backward_pool = PrioritizedTaskPool(self.backward, name=f"{self.name}_backward", **kwargs)
        self.dtype = backend_dtype if backend_dtype else self.module.input_layernorm.weight.dtype

    def inference_step(self, cache_metadata: torch.IntTensor, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with torch.inference_mode():
            attention_cache_handle = int(cache_metadata[0, 0].item())
            prefix_length = int(cache_metadata[0, 1].item())
            hidden_states = inputs[0]  # todo: in future, it would be best to support attention mask here
            assert (
                hidden_states.ndim == 3
            ), "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"

            with self.memory_cache.use_cache(attention_cache_handle) as cache:
                assert isinstance(self.module, BloomBlock) and cache.shape[0] == 2 and cache.ndim == 5
                layer_past = past_k, past_v = cache[0, :, :prefix_length], cache[1, :, :prefix_length]
                print("METADATA:", cache_metadata, past_k.shape, past_v.shape)
                hidden_states, (new_k, new_v) = self.module.forward(
                    hidden_states, layer_past=layer_past, use_cache=True
                )

                # todo remove these asserts once we pass all tests
                new_length = new_v.shape[1]
                assert new_length > prefix_length
                assert new_k.shape[0] == past_k.shape[0] and new_v.shape[0] == past_v.shape[0]
                assert new_k.shape[1] == new_length and new_v.shape[1] == new_length
                assert new_k.shape[2:] == past_k.shape[2:] and new_v.shape[2:] == past_v.shape[2:]
                cache[0, :, prefix_length:new_length, :] = new_k[:, prefix_length:new_length]
                cache[1, :, prefix_length:new_length, :] = new_v[:, prefix_length:new_length]
                return (hidden_states,)

    def get_pools(self) -> Sequence[TaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool
