"""
A prototype TensorParallel wrapper that works without torchrun. Original code by @BlackSamorez and @IaroslavLisniak .
This code is here temporarily, with authors' permission, until they make it publicly available otherwise.

The original code can be found here: https://github.com/BlackSamorez/petals_local_parallel , using MIT license
https://github.com/BlackSamorez/petals_local_parallel/blob/496e4a8ea641ff641e59309445ddc9fe0d7960cd/LICENCE
"""

import threading
from typing import List, Optional

import torch


class CollectiveOpetationBase:
    def __call__(self, x: torch.Tensor, rank: int):
        raise NotImplementedError()


class AllReduce(CollectiveOpetationBase):
    def __init__(self, world_size: int, reduce_op: callable = sum):
        self.scatter_reduce = ScatterReduce(world_size, reduce_op)
        self.all_gather = AllGather(world_size, gather_op=torch.cat)

    def __call__(self, x: torch.Tensor, rank: int):
        reduced_part = self.scatter_reduce(x, rank)
        return self.all_gather(reduced_part, rank).view_as(x)


class ScatterReduce(CollectiveOpetationBase):
    def __init__(self, world_size: int, reduce_op: callable = sum):
        self.world_size = world_size
        self.tensor_parts = [[] for _ in range(world_size)]
        self.parts_ready = [threading.Event() for _ in range(world_size)]
        self.reduce_op = reduce_op

    def __call__(self, x: torch.Tensor, rank: int):
        try:
            for i, part in enumerate(x.flatten().tensor_split(self.world_size)):
                self.tensor_parts[i].append(part)  # append is thread-safe. thanks, GIL!
                if len(self.tensor_parts[i]) == self.world_size:
                    self.parts_ready[i].set()  # can be called more than once; we dont care

            self.parts_ready[rank].wait()
            parts_to_reduce = [part.to(x.device, non_blocking=True)
                               for part in self.tensor_parts[rank]]
            reduced_part = self.reduce_op(parts_to_reduce)
            return reduced_part
        finally:
            # prepare for next forward; each rank clears its own data
            self.tensor_parts[rank].clear()
            self.parts_ready[rank].clear()


class AllGather(CollectiveOpetationBase):
    def __init__(self, world_size: int, gather_op: callable = torch.cat):
        self.world_size = world_size
        self.parts: List[Optional[torch.Tensor]] = [None for _ in range(world_size)]
        self.ranks_updated = []
        self.parts_ready = threading.Event()
        self.gather_op = gather_op

    def __call__(self, x: torch.Tensor, rank: int):
        parts, ranks_updated, parts_ready = self.parts, self.ranks_updated, self.parts_ready
        # ^-- note: we copy properties to locals so that the "finally" clause is thread-safe
        try:
            parts[rank] = x  # no race b/c each rank writes to a separate location
            ranks_updated.append(rank)  # append is thread-safe. thanks, GIL!
            if len(ranks_updated) == self.world_size:
                parts_ready.set()  # can be called more than once; we dont care
            parts_ready.wait()
            parts = [part.to(x.device, non_blocking=True) for part in parts]
            # note: for one of the parts with r == rank, part.to(device) is a no-op
            return self.gather_op(parts)
        finally:
            if ranks_updated[-1] == rank:
                self.parts = [None for _ in range(self.world_size)]
                self.ranks_updated = []
                self.parts_ready = threading.Event()
            # note: we can safely update these properties because all ranks have
            # copied self.parts_* to locals before passing parts_ready.wait
