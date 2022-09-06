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
from hivemind import use_hivemind_log_handler
from hivemind.utils import TensorDescriptor, get_logger

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)

Handle = int


class MemoryCache:
    """A shared cache for storing tensors that persist across calls. Main use case: storing past attention KVs"""

    def __init__(self, device: Union[str, torch.device], max_size_bytes: Optional[int]):
        self.max_size_bytes = max_size_bytes if max_size_bytes is not None else (2**64 - 1)
        self.device = device
        self._lock_metadata, self.size_decreased_event = mp.Lock(), mp.Event()
        self._current_size = mp.Value(ctypes.c_int64, 0, lock=False)
        self._handle_counter = mp.Value(ctypes.c_int64, 0, lock=False)
        self._active_handles: Optional[Dict[Handle, TensorDescriptor]] = None
        self._allocated_tensors: Optional[Dict[Handle, torch.Tensor]] = None
        self.runtime_pid = os.getpid()

        self._pipe_recv, self._pipe_send = mp.Pipe(duplex=False)  # any ConnectionHandler -> runtime
        self._pending_messages = mp.Value(ctypes.c_int64, 0, lock=False)
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
        allocated_handle = None
        allocated_size_bytes = descr.numel() * torch.finfo(descr.dtype).bits // 8
        loop = asyncio.get_event_loop()
        try:
            async with hivemind.utils.enter_asynchronously(self._lock_acquire_memory):
                if self.current_size_bytes + allocated_size_bytes > self.max_size_bytes:
                    await loop.run_in_executor(None, self._wait_until_available, allocated_size_bytes)
                async with hivemind.utils.enter_asynchronously(self._lock_metadata):
                    allocated_handle = int(self.handle_counter)
                    self.current_size_bytes += allocated_size_bytes
                    self.handle_counter += 1  # note: this will eventually overflow and it is okay
                    self._pending_messages.value += 1
                    self._pipe_send.send((allocated_handle, descr))

            yield allocated_handle
        finally:
            if allocated_handle is not None:
                async with hivemind.utils.enter_asynchronously(self._lock_metadata):
                    self._pending_messages.value += 1
                    self._pipe_send.send((allocated_handle, None))  # signal runtime to free that handle
                    self.current_size_bytes -= allocated_size_bytes
                self._memory_freed_event.set()

    def _wait_until_available(self, allocated_size_bytes: int, timeout: Optional[float] = None):
        # note: this function should only be called inside _lock_acquire_memory!
        if allocated_size_bytes > self.max_size_bytes:
            raise AllocationFailed(
                f"Could not allocate {allocated_size_bytes} bytes, max cache size = {self.max_size_bytes} bytes"
            )
        deadline = None if timeout is None else time.perf_counter() + timeout
        while self.current_size_bytes + allocated_size_bytes > self.max_size_bytes:
            remaining_time = deadline - time.perf_counter() if timeout is not None else None
            if not self._memory_freed_event.wait(remaining_time):
                raise AllocationFailed(f"Could not allocate {allocated_size_bytes} bytes in {timeout} seconds")
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
            if self._allocated_tensors is None:
                self._allocated_tensors = {}

            # read creation/deletion requests from connection handlers
            for i in range(int(self._pending_messages.value)):
                recv_handle, recv_data = self._pipe_recv.recv()
                self._pending_messages.value -= 1
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
