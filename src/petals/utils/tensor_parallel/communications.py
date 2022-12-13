"""
A prototype TensorParallel wrapper that works without torchrun. Original code by @BlackSamorez and @IaroslavLisniak .
This code is here temporarily, with authors' permission, until they make it publicly available otherwise.

The original code can be found here: https://github.com/BlackSamorez/petals_local_parallel , using MIT license
https://github.com/BlackSamorez/petals_local_parallel/blob/496e4a8ea641ff641e59309445ddc9fe0d7960cd/LICENCE
"""

import threading
from functools import partial
from typing import List, Optional, Sequence

import torch
from torch.nn.parallel import comm
from torch.nn.parallel._functions import ReduceAddCoalesced
from torch.nn.parallel._functions import Scatter, Gather, Broadcast


class CollectiveOpetationBase:
    def __call__(self, x: torch.Tensor, rank: int):
        raise NotImplementedError()


class AllReduce(CollectiveOpetationBase):
    def __init__(
        self, world_size: int, reduce_op: callable = comm.reduce_add, gather_op: callable = partial(comm.gather, dim=-1)
    ):
        self.scatter_reduce = ScatterReduce(world_size, reduce_op)
        self.all_gather = AllGather(world_size, gather_op, barrier=False)
        # note: AllGather does not need barrier here because scatter_reduce's ready event serves as barrier

    def __call__(self, x: torch.Tensor, rank: int):
        reduced_part = self.scatter_reduce(x, rank)
        return self.all_gather(reduced_part, rank).view_as(x)


class ScatterReduce(CollectiveOpetationBase):
    def __init__(self, world_size: int, reduce_op: callable = comm.reduce_add):
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
    def __init__(self, world_size: int, gather_op: callable = partial(comm.gather, dim=-1), barrier: bool = True):
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


def broadcast_coalesced(
    tensors: Sequence[torch.Tensor], devices: Sequence[torch.device], *, all_cuda: bool = None, **kwargs
) -> Sequence[Sequence[torch.Tensor]]:
    if all_cuda is None:
        all_cuda = all(device.type == 'cuda' for device in devices)
    if not all_cuda:
        broadcasted = [list() for _ in devices]
        for x in tensors:
            for i, device in enumerate(devices):
                broadcasted[i].append(x.to(device, non_blocking=True))
        return broadcasted
    flat_outputs = Broadcast.apply(devices, *tensors)
    return [flat_outputs[i * len(tensors): (i + 1) * len(tensors)] for i in range(len(devices))]


def gather(tensors: Sequence[torch.Tensor], dim: int = 0, destination: Optional[torch.device] = None, all_cuda: bool = None):
    """Gather tensors from multiple devices; differentiable w.r.t. input tensors"""
    if all_cuda is None:
        all_cuda = all(x.device.type=='cuda' for x in tensors)
    if destination is None:
        destination = tensors[0].device
    if not all_cuda:
        return torch.cat([x.to(destination, non_blocking=True) for x in tensors], dim=dim)
    return Gather.apply(destination, dim, tensors)



def reduce_add(tensors: Sequence[torch.Tensor], destination: Optional[torch.device] = None, all_cuda: bool = None):
    if all_cuda is None:
        all_cuda = all(x.device.type=='cuda' for x in tensors)
    if destination is None:
        destination = tensors[0].device
    if not all_cuda:
        return sum([tensor.to(destination, non_blocking=True) for tensor in tensors])
    return _ReduceAdd.apply(destination, tensors)


class _ReduceAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, destination: torch.device, *tensors: torch.Tensor):
        ctx.source_gpus = [tensor.get_device() for tensor in tensors]
        return comm.reduce_add(tensors, destination)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, ) + Broadcast.apply(ctx.source_gpus, *grad_outputs)
