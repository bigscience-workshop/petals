"""
A prototype TensorParallel wrapper that works without torchrun. Original code by @BlackSamorez and @IaroslavLisniak .
This code is here temporarily, with authors' permission, until they make it publicly available otherwise.

The original code can be found here: https://github.com/BlackSamorez/petals_local_parallel , using MIT license
https://github.com/BlackSamorez/petals_local_parallel/blob/496e4a8ea641ff641e59309445ddc9fe0d7960cd/LICENCE
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import comm, replicate

from petals.utils.tensor_parallel.communications import AllGather, AllReduce

Pattern, Arg = str, Union[int, str]
TensorAction = Union[str, Callable]


class SlicingConfig:
    def __init__(self, tensor_rules: Dict[Pattern, TensorAction], module_rules: Dict[Pattern, Dict[str, Any]]):
        self.tensor_rules = tensor_rules
        self.module_rules = module_rules

    @classmethod
    def get_default_config(cls, module: nn.Module) -> SlicingConfig:
        slicing_config = SlicingConfig({}, {})
        for name, module in module.named_modules():
            if isinstance(module, nn.Linear):
                slicing_config.tensor_rules[name + ".(weight|bias)"] = "vertical"
                slicing_config.module_rules[name] = {"input": {}, "output": {0: "gather"}, "attributes": {}}
        return slicing_config


def slice_weight_vertical(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return tensor.tensor_split(world_size, dim=-2)[rank]


def slice_bias_vertical(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return tensor.tensor_split(world_size)[rank]


def slice_weight_horizontal(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return tensor.tensor_split(world_size, dim=-1)[rank]


def slice_bias_horizontal(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return tensor / world_size


SLICING_RULES = {
    ("vertical", "weight"): slice_weight_vertical,
    ("vertical", "bias"): slice_bias_vertical,
    ("horizontal", "weight"): slice_weight_horizontal,
    ("horizontal", "bias"): slice_bias_horizontal,
}


def slice_tensors_(
    key_parameter_iterator: Iterator[Tuple[str, nn.Parameter]], tensor_rules: Dict[Arg, str], rank: int, world_size: int
):
    regular_rules = [(re.compile(key), value) for key, value in tensor_rules.items()]

    with torch.no_grad():
        for name, param in key_parameter_iterator:
            for pattern, rule in regular_rules:
                if pattern.search(name) is not None:
                    name_ending = name.split(".")[-1]
                    param.data = SLICING_RULES[rule, name_ending](param.data, rank=rank, world_size=world_size).clone()


def process_input(rules: Dict[Arg, str], rank: int, world_size: int, *args, **kwargs):
    extended_kwargs = dict(kwargs)
    extended_kwargs.update(enumerate(args))
    for target, action in rules.items():
        if not isinstance(extended_kwargs.get(target), torch.Tensor):
            continue  # optional parameter is None or False
        action_type, *maybe_dim = action
        if action_type == "cut":
            extended_kwargs[target] = extended_kwargs[target].tensor_split(world_size, dim=maybe_dim)[rank]
        elif action_type == "scale":
            extended_kwargs[target] = extended_kwargs[target] / world_size
        else:
            raise Exception(f"unexpected action {action_type}")

    args = [extended_kwargs.pop(i) for i in range(len(args))]
    return args, extended_kwargs


def process_output(output, rules: Dict[Arg, Callable[[torch.Tensor, int], torch.Tensor]], rank: int):
    if isinstance(output, torch.Tensor):
        return process_output([output], rules, rank)[0]
    for target, action in rules.items():
        output[target] = action(output[target], rank)
    return output


def process_attr(module: nn.Module, rules: Dict[Arg, str], rank: int, world_size: int):
    for attr, action in rules.items():
        if action == "scale_int":
            setattr(module, attr, getattr(module, attr) // world_size)
        else:
            raise NotImplementedError(action)


class ParallelLayerWrapper(nn.Module):
    def __init__(self, module: nn.Module, module_rules: dict, rank: int, world_size: int):
        super().__init__()
        self.module = module
        process_attr(self.module, module_rules["attributes"], rank=rank, world_size=world_size)

        self.input_rules = module_rules["input"]
        self.output_rules = module_rules["output"]

        self.rank = rank
        self.world_size = world_size

    def forward(self, *args, **kwargs):
        args, kwargs = process_input(self.input_rules, self.rank, self.world_size, *args, **kwargs)
        output = self.module(*args, **kwargs)
        return process_output(output, self.output_rules, self.rank)


def create_collective_ops(module_rules: dict, devices: Sequence[torch.device]):
    world_size = len(devices)
    if any(device.type == "cpu" for device in devices):
        reduce_op = lambda xs, destination: sum(x.to(destination) for x in xs)
        gather_op = lambda xs, destination: torch.cat([x.to(destination) for x in xs], dim=-1)
    else:
        reduce_op = comm.reduce_add  # gpu-optimized ops
        gather_op = lambda xs, destination: comm.gather(xs, dim=-1, destination=destination)

    unique_output_transforms = {op for rules in module_rules.values() for op in rules["output"].values()}
    transform_map = {}
    for transform in unique_output_transforms:
        if transform == "sum":
            transform_map[transform] = AllReduce(world_size, reduce_op, gather_op)
        elif transform == "gather":
            transform_map[transform] = AllGather(world_size, gather_op)
        elif callable(transform):
            transform_map[transform] = transform  # user-defined transform, no action needed
        else:
            raise NotImplementedError(f"Unknown output transform {transform}")

    initialized_module_rules = {}
    for pattern, rules in module_rules.items():
        output_ops = {key: transform_map[rule] for key, rule in rules["output"].items()}
        initialized_module_rules[pattern] = dict(rules, output=output_ops)
    return initialized_module_rules


def wrap_submodules_(model: nn.Module, module_rules: dict, rank: int, world_size: int):
    unique_wrappers = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            for pattern, rule in module_rules.items():
                if re.search(pattern, name) is not None:
                    unique_wrappers[module] = ParallelLayerWrapper(module, rule, rank=rank, world_size=world_size)

    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if child in unique_wrappers:
                setattr(parent, child_name, unique_wrappers[child])


def create_module_shard(
    module: nn.Module, device: torch.device, config: SlicingConfig, rank: int, world_size: int
) -> nn.Module:
    if device.type == "cuda":
        (replica,) = replicate(module, (device,), detach=True)
    else:
        replica = deepcopy(module).to(device)
    slice_tensors_(replica.named_parameters(), config.tensor_rules, rank, world_size)
    wrap_submodules_(nn.ModuleList([replica]), config.module_rules, rank, world_size)
    return replica
