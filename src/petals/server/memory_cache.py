"""
A pytorch memory cache that can be allocated by ConnectionHandler (on cpu) and used over multiple calls to Runtime.

For now, the only purpose of this code is to ensure that allocated memory will be deleted properly.

"""
import asyncio
import contextlib
import ctypes
import multiprocessing as mp
import os
import time
from typing import AsyncContextManager, Dict, Optional, Union

import hivemind
import torch
from hivemind.utils import TensorDescriptor, get_logger

from petals.utils.asyncio import shield_and_wait

logger = get_logger(__file__)

Handle = int


class MemoryCache:
    """A shared cache for storing tensors that persist across calls. Main use case: storing past attention KVs"""

    def __init__(self, device: Union[str, torch.device], max_size_bytes: Optional[int], alloc_timeout: float):
        self.max_size_bytes = max_size_bytes if max_size_bytes is not None else (2**64 - 1)
        self.alloc_timeout = alloc_timeout
        self.device = device
        self._lock_metadata, self.size_decreased_event = mp.Lock(), mp.Event()
        self._current_size = mp.Value(ctypes.c_int64, 0, lock=False)
        self._handle_counter = mp.Value(ctypes.c_int64, 0, lock=False)
        self._allocated_tensors: Dict[Handle, torch.Tensor] = {}
        self.runtime_pid = os.getpid()

        self._pipe_recv, self._pipe_send = mp.Pipe(duplex=False)  # any ConnectionHandler -> runtime
        self._lock_acquire_memory = mp.Lock()
        self._memory_freed_event = mp.Event()

    @property
    def current_size_bytes(self) -> int:
        return self._current_size.value

    @current_size_bytes.setter
    def current_size_bytes(self, value: int):
        self._current_size.value = value

    @property
    def handle_counter(self) -> int:
        return self._handle_counter.value

    @handle_counter.setter
    def handle_counter(self, value: int):
        self._handle_counter.value = value

    @contextlib.asynccontextmanager
    async def allocate_cache(self, descr: TensorDescriptor) -> AsyncContextManager[Handle]:
        """
        Create a handle that is associated with buffers on unique device. If cache full, raises AllocationFailed.

        :param descr: allocate a tensor of this size, dtype, etc

        :note: This function should be called by connection handlers, it can be called concurrently from multiple processes.
        Furthermore, it can be called concurrently with at most one use_cache call in runtime.
        """
        assert os.getpid() != self.runtime_pid, "must be called by a ConnectionHandler, not runtime"
        assert descr.device is None and descr

        alloc_size = descr.numel() * torch.finfo(descr.dtype).bits // 8
        alloc_task = asyncio.create_task(self._schedule_alloc(alloc_size, descr))
        try:
            yield await shield_and_wait(alloc_task)
        finally:
            await shield_and_wait(self._schedule_free(alloc_size, alloc_task))

    async def _schedule_alloc(self, alloc_size: int, descr: TensorDescriptor) -> Handle:
        """
        This method should be called inside asyncio.shield() because:
            - hivemind.utils.enter_asynchronously() does not always release the lock on cancellation
        """

        loop = asyncio.get_event_loop()
        async with hivemind.utils.enter_asynchronously(self._lock_acquire_memory):
            if self.current_size_bytes + alloc_size > self.max_size_bytes:
                await loop.run_in_executor(None, self._wait_until_available, alloc_size, self.alloc_timeout)
            async with hivemind.utils.enter_asynchronously(self._lock_metadata):
                handle = int(self.handle_counter)
                self.current_size_bytes += alloc_size
                self.handle_counter += 1  # note: this will eventually overflow and it is okay
                self._pipe_send.send((handle, descr))
                return handle

    async def _schedule_free(self, alloc_size: int, alloc_task: asyncio.Task):
        """
        This method should be called inside asyncio.shield() because:
            - hivemind.utils.enter_asynchronously() does not always release the lock on cancellation
            - _schedule_free() must finish freeing memory even in case of cancellation
        """

        if alloc_task.exception() is not None:
            return
        handle = alloc_task.result()

        async with hivemind.utils.enter_asynchronously(self._lock_metadata):
            self._pipe_send.send((handle, None))  # signal runtime to free that handle
            self.current_size_bytes -= alloc_size
        self._memory_freed_event.set()

    def _wait_until_available(self, allocated_size: int, timeout: Optional[float] = None):
        # note: this function should only be called inside _lock_acquire_memory!
        if allocated_size > self.max_size_bytes:
            raise AllocationFailed(
                f"Could not allocate {allocated_size} bytes, max cache size = {self.max_size_bytes} bytes"
            )
        deadline = None if timeout is None else time.perf_counter() + timeout
        while self.current_size_bytes + allocated_size > self.max_size_bytes:
            remaining_time = deadline - time.perf_counter() if timeout is not None else None
            if not self._memory_freed_event.wait(remaining_time):
                raise AllocationFailed(
                    f"Server's attention cache is full, failed to allocate {allocated_size} bytes in {timeout} seconds"
                )
            self._memory_freed_event.clear()

    @contextlib.contextmanager
    def use_cache(self, handle: Handle) -> torch.Tensor:
        """
        Return a tensor that was previously allocated with try_allocate_cache,

        :note: This method is called by ExpertBackend in runtime: a single process with NO process parallelism.
        However, runtime may call use_cache concurrently with one or more connection handlers calling allocate_cache
        """
        assert os.getpid() == self.runtime_pid
        # note: this specific function is not concurrent, so you can safely allocate/offload/defragment data here

        with self._lock_metadata:
            # read creation/deletion requests from connection handlers
            while self._pipe_recv.poll():
                recv_handle, recv_data = self._pipe_recv.recv()
                if isinstance(recv_data, TensorDescriptor):
                    self._allocated_tensors[recv_handle] = recv_data.make_zeros(device=self.device)
                elif recv_data is None:
                    if recv_handle not in self._allocated_tensors:
                        logger.warning(
                            f"Sanity check failed: asked to delete handle {recv_handle}, but there is no such handle"
                        )
                    self._allocated_tensors.pop(recv_handle, None)
                else:
                    logger.error(f"MemoryCache pipe received unexpected message: {recv_data}")

        assert handle in self._allocated_tensors, f"Sanity check failed: no such handle ({handle})"
        yield self._allocated_tensors[handle]


class AllocationFailed(Exception):
    pass
