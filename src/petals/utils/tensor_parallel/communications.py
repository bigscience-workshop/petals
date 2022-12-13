"""
A prototype TensorParallel wrapper that works without torchrun. Original code by @BlackSamorez and @IaroslavLisniak .
This code is here temporarily, with authors' permission, until they make it publicly available otherwise.

The original code can be found here: https://github.com/BlackSamorez/tensor_parallel , using MIT license
https://github.com/BlackSamorez/tensor_parallel/blob/496e4a8ea641ff641e59309445ddc9fe0d7960cd/LICENCE
"""
from __future__ import annotations

import threading
from typing import List, Optional

import torch
import petals.utils.tensor_parallel.cross_device_ops as cross_device_ops


class CollectiveOpetationBase:
    def __call__(self, x: torch.Tensor, rank: int):
        raise NotImplementedError()


class AllReduce(CollectiveOpetationBase):
    def __init__(
        self, world_size: int, reduce_op: callable = cross_device_ops.reduce_add, gather_op: callable = cross_device_ops.gather
    ):
        self.scatter_reduce = ScatterReduce(world_size, reduce_op)
        self.all_gather = AllGather(world_size, gather_op, barrier=False)
        # note: AllGather does not need barrier here because scatter_reduce's ready event serves as barrier

    def __call__(self, x: torch.Tensor, rank: int):
        reduced_part = self.scatter_reduce(x, rank)
        return self.all_gather(reduced_part, rank).view_as(x)


class ScatterReduce(CollectiveOpetationBase):
    def __init__(self, world_size: int, reduce_op: callable = cross_device_ops.reduce_add):
        self.world_size = world_size
        self.tensor_parts = [[] for _ in range(world_size)]
        self.parts_ready = [threading.Event() for _ in range(world_size)]
        self.reduce_op = reduce_op

    def __call__(self, x: torch.Tensor, rank: int):
        try:
            for i, part in enumerate(x.flatten().tensor_split(self.world_size)):
                self.tensor_parts[i].append(part)  # append is thread-safe. thanks, GIL!
                if len(self.tensor_parts[i]) == self.world_size:
                    self.parts_ready[i].set()  # can be called more than once; we don't care

            self.parts_ready[rank].wait()
            reduced_part = self.reduce_op(self.tensor_parts[rank], x.device)
            return reduced_part
        finally:
            # prepare for next forward; each rank clears its own data
            self.tensor_parts[rank].clear()
            self.parts_ready[rank].clear()


class AllGather(CollectiveOpetationBase):
    def __init__(self, world_size: int, gather_op: callable = cross_device_ops.gather, barrier: bool = True):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size) if barrier else None
        self.parts: List[Optional[torch.Tensor]] = [None for _ in range(world_size)]
        self.ranks_updated = []
        self.parts_ready = threading.Event()
        self.gather_op = gather_op

    def __call__(self, x: torch.Tensor, rank: int):
        if self.barrier is not None:
            self.barrier.wait()  # if this code is ran multiple times in quick succession,
        # this even will wait for the previous call to finish before starting a new one
        parts, ranks_updated, parts_ready = self.parts, self.ranks_updated, self.parts_ready
        # ^-- note: we copy properties to locals so that the "finally" clause is thread-safe
        try:
            parts[rank] = x  # no race b/c each rank writes to a separate location
            ranks_updated.append(rank)  # append is thread-safe. thanks, GIL!
            if len(ranks_updated) == self.world_size:
                parts_ready.set()  # can be called more than once; we dont care
            parts_ready.wait()
            # note: for one of the parts with r == rank, part.to(device) is a no-op
            return self.gather_op(parts, x.device)
        finally:
            if ranks_updated[-1] == rank:
                self.parts = [None for _ in range(self.world_size)]
                self.ranks_updated = []
                self.parts_ready = threading.Event()
            # note: we can safely update these properties because all ranks have
            # copied self.parts_* to locals before passing parts_ready.wait
