import asyncio
import multiprocessing as mp
import random
import time
from typing import Optional

import pytest
import pytest_asyncio  # make sure the module exists; otherwise the test will be skipped
import torch
from hivemind import TensorDescriptor

from petals.server.memory_cache import AllocationFailed, MemoryCache
from petals.utils.misc import get_size_in_bytes


def _make_tensor_descriptor(num_bytes: int, dtype: Optional[torch.dtype] = None):
    if dtype is None:
        dtype = random.choice((torch.int64, torch.int8, torch.uint8, torch.float32, torch.bfloat16, torch.bool))
    elem_size_bytes = get_size_in_bytes(dtype)
    descr = TensorDescriptor.from_tensor(torch.empty((num_bytes // elem_size_bytes,), dtype=dtype))
    return descr


@pytest.mark.asyncio
async def test_cache_timeout():
    cache = MemoryCache(max_size_bytes=1024, max_alloc_timeout=0.5)
    cache.runtime_pid += 1  # pretend we're another process
    async with cache.allocate_cache(_make_tensor_descriptor(768), timeout=0):
        pass

    async with cache.allocate_cache(_make_tensor_descriptor(100), timeout=999):
        async with cache.allocate_cache(_make_tensor_descriptor(512), timeout=0):
            async with cache.allocate_cache(_make_tensor_descriptor(128), _make_tensor_descriptor(32), timeout=1):
                t_start = time.perf_counter()
                with pytest.raises(AllocationFailed):
                    async with cache.allocate_cache(_make_tensor_descriptor(768), timeout=0.1):
                        pass
                assert 0.1 < time.perf_counter() - t_start < 0.2, "wait time exceeds alloc timeout"
                async with cache.allocate_cache(_make_tensor_descriptor(128), timeout=float("inf")):
                    pass

                t_start = time.perf_counter()
                with pytest.raises(AllocationFailed):
                    async with cache.allocate_cache(_make_tensor_descriptor(384), timeout=1.0):  # exceeds max timeout
                        pass
                assert 0.5 < time.perf_counter() - t_start < 0.6, "wait time exceeds max alloc timeout"

            # test memory allocation when another task frees the memory
            async def _klog_the_cache():
                async with cache.allocate_cache(_make_tensor_descriptor(512), timeout=0.2):
                    pass

            large_alloc_task = asyncio.create_task(_klog_the_cache())

            t_start = time.perf_counter()
            await asyncio.sleep(0.05)  # wait for large alloc to enqueue
            async with cache.allocate_cache(_make_tensor_descriptor(128), timeout=float("inf")):  # exceeds max timeout
                pass  # this memory should allocate once the background task clears the queue
            assert 0.2 < time.perf_counter() - t_start < 0.3, "memory should be allocated after background task clears"
            with pytest.raises(AllocationFailed):
                await large_alloc_task

            # test that zero-timeout allocation fails instantaneously even if someone else is awaiting alloc
            large_alloc_task = asyncio.create_task(_klog_the_cache())
            t_start = time.perf_counter()
            await asyncio.sleep(0.05)  # wait for large alloc to enqueue
            with pytest.raises(AllocationFailed):
                async with cache.allocate_cache(_make_tensor_descriptor(512), timeout=0):
                    pass  # this memory should allocate once the background task clears the queue
            assert time.perf_counter() - t_start < 0.1, "zero-timeout task should fail (or succeed) instantaneously"
            with pytest.raises(AllocationFailed):
                await large_alloc_task


@pytest.mark.asyncio
async def test_unlimited_timeout():
    cache = MemoryCache(max_size_bytes=1024)
    cache.runtime_pid += 1  # pretend we're another process
    t_start = time.perf_counter()

    async def _klog_the_cache():
        async with cache.allocate_cache(_make_tensor_descriptor(512), timeout=0.2):
            await asyncio.sleep(0.5)

    alloc_task = asyncio.create_task(_klog_the_cache())
    await asyncio.sleep(0.1)
    async with cache.allocate_cache(_make_tensor_descriptor(768), timeout=float("inf")):
        await alloc_task
    assert 0.5 < time.perf_counter() - t_start < 0.6, "memory should be allocated after background task clears"


@pytest.mark.asyncio
async def test_cache_usage():
    cache = MemoryCache(max_size_bytes=2048)
    alloc_event, dealloc_a_event, dealloc_bcd_event, dealloc_e_event, dealloc_f_event = (mp.Event() for _ in range(5))
    pipe_receiver, pipe_sender = mp.Pipe(duplex=False)
    with pytest.raises(AssertionError):
        async with cache.allocate_cache(_make_tensor_descriptor(123), timeout=1):
            pass  # fails because cache must be allocated from another process

    descr_a = TensorDescriptor.from_tensor(torch.empty(768, dtype=torch.uint8))  # 768 bytes
    descr_b = TensorDescriptor.from_tensor(torch.empty((), dtype=torch.float64))  # 8 bytes
    descr_c = TensorDescriptor.from_tensor(torch.empty((33,), dtype=torch.bool))  # 33 bytes
    descr_d = TensorDescriptor.from_tensor(torch.empty((0,), dtype=torch.int64))  # 0 bytes
    descr_e = TensorDescriptor.from_tensor(torch.empty((96, 8), dtype=torch.bfloat16))  # 1536 bytes
    descr_f = TensorDescriptor.from_tensor(torch.empty((1792,), dtype=torch.uint8))  # 1792 bytes

    async def _allocate_and_wait(dealloc_event, *descrs, timeout=None):
        loop = asyncio.get_event_loop()
        async with cache.allocate_cache(*descrs, timeout=timeout) as handles:
            pipe_sender.send(handles)
            await loop.run_in_executor(None, dealloc_event.wait)

    async def _allocate_af():
        alloc_event.wait()
        allocate_a_task = asyncio.create_task(_allocate_and_wait(dealloc_a_event, descr_a))
        await allocate_a_task
        allocate_f_task = asyncio.create_task(_allocate_and_wait(dealloc_f_event, descr_f))  # klogs the cache
        await allocate_f_task

    alloc_process1 = mp.context.ForkProcess(target=lambda: asyncio.run(_allocate_af()), daemon=True)
    alloc_process1.start()

    async def _allocate_bcde():
        alloc_event.wait()
        await asyncio.sleep(0.1)  # ensure that the other tensor is always allocated (and sent through pipe) first
        allocate_bcd_task = asyncio.create_task(_allocate_and_wait(dealloc_bcd_event, descr_b, descr_c, descr_d))
        allocate_e_task = asyncio.create_task(_allocate_and_wait(dealloc_e_event, descr_e))  # doesn't fit
        await asyncio.wait({allocate_e_task, allocate_bcd_task}, return_when=asyncio.ALL_COMPLETED)

    alloc_process2 = mp.context.ForkProcess(target=lambda: asyncio.run(_allocate_bcde()), daemon=True)
    alloc_process2.start()
    assert cache.current_size_bytes == 0
    alloc_event.set()
    (handle_a,) = pipe_receiver.recv()

    handle_b, handle_c, handle_d = pipe_receiver.recv()

    with cache.use_cache(handle_a) as (tensor_a,):
        assert tensor_a.dtype == torch.uint8
        tensor_a[2:5] = torch.tensor((42, 43, 44))

    with cache.use_cache(handle_a, handle_b, handle_d) as (tensor_a, tensor_b, tensor_d):
        assert tensor_b.dtype == torch.float64 and tensor_b.numel() == 1 and tensor_b.ndim == 0
        assert tensor_d.dtype == torch.int64 and tensor_d.numel() == 0
        tensor_a += 1
        tensor_b[...] = -1.337
    assert cache.current_size_bytes == 809  # this checks a,b,c,d are allocated but b still awaits memory

    dealloc_bcd_event.set()
    await asyncio.sleep(0.1)
    assert cache.current_size_bytes == 768  # only tensor a should be allocated
    with pytest.raises(KeyError):
        with cache.use_cache(handle_a, handle_b):
            pass  # one of handles (c) is deallocated
    with pytest.raises(KeyError):
        with cache.use_cache(handle_d):
            pass  # handle_d is deallocated correctly, even though it is never used
    with cache.use_cache(handle_a) as (tensor_a,):
        assert tuple(tensor_a[2:5]) == (43, 44, 45)

    dealloc_a_event.set()
    (handle_e,) = pipe_receiver.recv()  # e can finally be allocated
    await asyncio.sleep(0.1)
    assert cache.current_size_bytes == 1536  # tensor e should finally be able to allocate

    with pytest.raises(KeyError):
        with cache.use_cache(handle_a):
            pass  # tensor a is no longer allocated
    with cache.use_cache(handle_e) as (tensor_e,):
        assert tensor_e.dtype == torch.bfloat16 and tensor_e.shape == (96, 8)

    dealloc_e_event.set()
    await asyncio.sleep(0.1)
    assert cache.current_size_bytes == 1792  # only tensor f is still allocated
    dealloc_f_event.set()

    alloc_process1.join()
    alloc_process2.join()
    await asyncio.sleep(0.1)
    assert cache.current_size_bytes == 0
    assert cache.current_size_bytes == 0
    assert alloc_process1.exitcode == 0, "allocation process 1 failed or did not finish, see stderr for details"
    assert alloc_process2.exitcode == 0, "allocation process 2 failed or did not finish, see stderr for details"
