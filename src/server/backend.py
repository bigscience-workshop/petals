"""Code for serving bloom blocks via hivemind-server"""
import ctypes
import multiprocessing as mp
import os
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue
from typing import Any, Dict, Optional, Sequence, Tuple, List

import torch
from hivemind import BatchTensorDescriptor, use_hivemind_log_handler
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.moe.server.task_pool import Task, TaskPool
from hivemind.utils import InvalidStateError, get_logger

from src.bloom.from_pretrained import BloomBlock
from src.server.cache import MemoryCache
from src.utils.misc import is_dummy

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
        self.inference_schema = (
            (
                *self.args_schema,
                BatchTensorDescriptor((), dtype=self.dtype),
                BatchTensorDescriptor((), dtype=torch.int64),
            ),
            self.kwargs_schema,
        )

    def inference_step(self, cache_metadata: torch.IntTensor, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with torch.inference_mode():
            attention_cache_handle = int(cache_metadata[0, 0].item())
            prefix_length = int(cache_metadata[0, 1].item())
            (hidden_states, hypo_ids) = inputs
            assert (
                hidden_states.ndim == 3
            ), "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"

            with self.memory_cache.use_cache(attention_cache_handle) as cache:
                assert isinstance(self.module, BloomBlock) and cache.shape[0] == 2 and cache.ndim == 5
                if not is_dummy(hypo_ids):
                    cache[:, :] = cache[:, hypo_ids]  # in-place reorder cache by hypo ids
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

    def get_info(self) -> Dict[str, Any]:
        """Get expert parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(super().get_info(), inference_schema=self.inference_schema)
