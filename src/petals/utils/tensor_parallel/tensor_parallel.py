import logging
from typing import Any, List, Optional, Sequence, Union

import torch
from hivemind import get_logger, nested_flatten, nested_pack, use_hivemind_log_handler
from torch import nn
from torch._utils import _get_all_device_indices, _get_device_index
from torch.nn.parallel import parallel_apply

from petals.utils.tensor_parallel.communications import broadcast_coalesced
from petals.utils.tensor_parallel.slicer_wrapper import SlicingConfig, create_collective_ops, create_module_shard

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class TensorParallel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        device_ids: Optional[Sequence[torch.device]] = None,
        output_device: Optional[torch.device] = None,
        config: Optional[SlicingConfig] = None,
    ):
        if isinstance(module, nn.Linear):
            # if linear is the only module, it interferes with internal slicing rules
            module = nn.Sequential(module)
        super().__init__()
        self.module_shards: List[nn.Module] = nn.ModuleList()
        if device_ids is None:
            device_ids = _get_all_device_indices()
        if device_ids is None or len(device_ids) <= 1:
            self.module_shards.append(module)
            self.device_ids = []
            return

        self.devices = tuple(torch.device(d) for d in device_ids)
        self.device_ids = [_get_device_index(x, optional=True, allow_cpu=True) for x in device_ids]
        self.output_device_index = self.devices.index(output_device) if output_device is not None else 0
        world_size = len(self.devices)

        if config is None:
            config = SlicingConfig.get_default_config(module)
            logger.info("Using automatic config: apply TP to linear layers only")

        module_rules_with_ops = create_collective_ops(config.module_rules, self.devices)
        # ^-- creates collective op instances, such as AllReduce and AllGather
        config_with_ops = SlicingConfig(config.tensor_rules, module_rules_with_ops)

        for rank, device in enumerate(self.devices):
            self.module_shards.append(create_module_shard(module, device, config_with_ops, rank, world_size))

        # self-diagnostics: check if the model was sharded properly
        original_params = sum(p.numel() for p in module.parameters())
        params_per_shard = [sum(p.numel() for p in shard.parameters()) for shard in self.module_shards]
        assert sum(params_per_shard) >= original_params, "Internal assert failed: lost some parameters during sharding"
        self.param_fractions = tuple(params_i / original_params for params_i in params_per_shard)
        inefficiency_rate = (sum(self.param_fractions) - 1) / len(device_ids)  # extra params rate per GPU
        log_level = logging.DEBUG if inefficiency_rate < 0.1 else logging.WARNING
        logger.log(
            log_level,
            f"Inefficiency warning: model has {original_params} but shards have {params_per_shard} params. "
            f"This means that each GPU uses {inefficiency_rate * 100:.3f}% extra memory for parameters",
        )

    def _replicate_one_input(self, inp: Union[torch.Tensor, Any]):
        if isinstance(inp, torch.Tensor):
            return [inp.to(device) for device in self.device_ids]

    def forward(self, *args, **kwargs):
        if len(self.module_shards) <= 1:
            return self.module_shards[0](*args, **kwargs)
        args_and_kwargs = (args, kwargs)
        flat_tensors = [obj for obj in nested_flatten(args_and_kwargs) if isinstance(obj, torch.Tensor)]
        iter_flat_tensors_replicated = iter(broadcast_coalesced(flat_tensors, self.devices))
        args_and_kwargs_replicated = [list() for _ in self.device_ids]
        for obj in nested_flatten(args_and_kwargs):
            if isinstance(obj, torch.Tensor):
                tensors_replicated = next(iter_flat_tensors_replicated)
                for idx in range(len(self.module_shards)):
                    args_and_kwargs_replicated[idx].append(tensors_replicated[idx])
            else:
                for idx in range(len(self.module_shards)):
                    args_and_kwargs_replicated[idx].append(obj)
        for idx in range(len(self.module_shards)):
            args_and_kwargs_replicated[idx] = nested_pack(args_and_kwargs_replicated[idx], args_and_kwargs)
        inputs, kwargs_tup = zip(*args_and_kwargs_replicated)
        return parallel_apply(self.module_shards, inputs, kwargs_tup=kwargs_tup)[self.output_device_index]
