import contextlib
import ctypes
import multiprocessing as mp
from typing import Dict, Tuple

import torch


class MemoryCache:
    lock: mp.Lock
    runtime_pid: int
    handle_counter: mp.Value[ctypes.c_uint64]
    current_size: mp.Value[ctypes.c_uint64]
    _runtime_data: Dict[int, SomeKindOfTupleWithTensors]  # workaround for now, while we are on CPU

    @contextlib.asynccontextmanager
    async def allocate_cache(self, size: torch.Size, dtype: torch.dtype) -> Optional[int]:
        """
        Allocate buffers for attention cache on the compute device, return a unique handle;
        This function should be called by connection handler processes, may be called concurrently
        """
        assert os.getpid() != self.runtime_pid
        try:
            async with acquire_asynchronously(self.lock):
                check_and_update_size(current_size, size, dtype)
                if enough_space:
                    self.handle_counter.value += 1
                    handle = int(self.handle_counter.value)
                    # note: you cannot allocate data here because this is
                    TODO_SOMEHOW_COMUNICATE_WITH_RUNTIME_TO_CREATE_THE_RIGHT_DATA
            yield handle
        finally:
            todo_deallocate(self, handle)
            # ^-- this should NOT move any data. But it may mark data for movement during next allocation
            self.data.pop(handle, None);

    def use_cache(self, handle: int) -> Tuple[mp.Value, torch.Tensor, torch.Tensor]:
        """Return a previously allocated cache, called by ExpertBackend in runtime (a single process)"""
        assert os.getpid() == self.runtime_pid
        with self.lock:
            if first_time:
                allocate_stuff(self._runtime_data)
            yield self.data[handle]
