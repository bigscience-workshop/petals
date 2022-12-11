from typing import Optional, Sequence

import torch
from torch import nn
from torch._utils import _get_all_device_indices, _get_available_device_type, _get_device_index
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.data_parallel import _check_balance

from petals.utils.tensor_parallel.slicer_wrapper import SlicingConfig


class TensorParallel(nn.Module):
    def __init__(
            self, module: nn.Module,
            device_ids:Optional[Sequence[torch.device]]=None,
            output_device: Optional[torch.device]=None,
            config: Optional[SlicingConfig]=None,
    ):
        super().__init__()
        self.module_shards = nn.ModuleList()
        device_type = _get_available_device_type()
        if device_type is None:
            self.module_shards.append(module)
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = _get_all_device_indices()

        if output_device is None:
            output_device = device_ids[0]

        self.device_ids = [_get_device_index(x, True) for x in device_ids]
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])
        self.world_size = len(self.device_ids)

        _check_balance(self.device_ids)

        config_with_collective_ops = MAKE_COLLECTIVE_OPS(config)

        for rank, device in enumerate(self.device_ids):
            replica = replicate(self.module, (device,), detach=True)
            TODO_MAKE_SHARD(replica, config_with_collective_ops, rank, world_size)

    def forward(self, *args, **kwargs):
        def scatter_map(obj):
            if isinstance(obj, torch.Tensor):
                return [obj.clone().to(targets) for targets in self.devices]
            if isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields"):
                return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
            if isinstance(obj, tuple) and len(obj) > 0:
                return list(zip(*map(scatter_map, obj)))
            if isinstance(obj, list) and len(obj) > 0:
                return [list(i) for i in zip(*map(scatter_map, obj))]
            if isinstance(obj, dict) and len(obj) > 0:
                return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
            return [obj for _ in self.devices]

        inputs = scatter_map(args)
        kwargs_tup = scatter_map(kwargs)

        return parallel_apply(self.slices, inputs, kwargs_tup=kwargs_tup)[0]
