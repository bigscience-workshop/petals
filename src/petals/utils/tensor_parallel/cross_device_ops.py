"""
A prototype TensorParallel wrapper that works without torchrun. Original code by @BlackSamorez and @IaroslavLisniak .
This code is here temporarily, with authors' permission, until they make it publicly available otherwise.

The original code can be found here: https://github.com/BlackSamorez/tensor_parallel , using MIT license
https://github.com/BlackSamorez/tensor_parallel/blob/496e4a8ea641ff641e59309445ddc9fe0d7960cd/LICENCE
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch.nn.parallel import comm
from torch.nn.parallel._functions import Broadcast, Gather


def broadcast_coalesced(
    tensors: Sequence[torch.Tensor], devices: Sequence[torch.device], *, all_cuda: bool = None, **kwargs
) -> Sequence[Sequence[torch.Tensor]]:
    if all_cuda is None:
        all_cuda = all(device.type == "cuda" for device in devices)
    if not all_cuda:
        broadcasted = [list() for _ in devices]
        for x in tensors:
            for i, device in enumerate(devices):
                broadcasted[i].append(x.to(device, non_blocking=True))
        return broadcasted
    flat_outputs = Broadcast.apply(devices, *tensors)
    return [flat_outputs[i * len(tensors) : (i + 1) * len(tensors)] for i in range(len(devices))]


def gather(
    tensors: Sequence[torch.Tensor], dim: int = 0, destination: Optional[torch.device] = None, all_cuda: bool = None
):
    """Gather tensors from multiple devices; differentiable w.r.t. input tensors"""
    if all_cuda is None:
        all_cuda = all(x.device.type == "cuda" for x in tensors)
    if destination is None:
        destination = tensors[0].device
    if not all_cuda:
        return torch.cat([x.to(destination, non_blocking=True) for x in tensors], dim=dim)
    return Gather.apply(destination, dim, *tensors)


def reduce_add(tensors: Sequence[torch.Tensor], destination: Optional[torch.device] = None, all_cuda: bool = None):
    if all_cuda is None:
        all_cuda = all(x.device.type == "cuda" for x in tensors)
    if destination is None:
        destination = tensors[0].device
    if not all_cuda:
        return sum([tensor.to(destination, non_blocking=True) for tensor in tensors])
    return _ReduceAdd.apply(destination, *tensors)


class _ReduceAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, destination: torch.device, *tensors: torch.Tensor):
        ctx.source_gpus = [tensor.get_device() for tensor in tensors]
        return comm.reduce_add(tensors, destination)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) + Broadcast.apply(ctx.source_gpus, *grad_outputs)
