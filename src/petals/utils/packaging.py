import re
from typing import Any

import torch
from hivemind import nested_flatten, nested_pack

# TODO: Move functions to hivemind


def mark_masked_tensor(index: int):
    return b"__T" + str(index).encode()


def is_masked_tensor(item: Any):
    return isinstance(item, bytes) and re.match(b"^__T\d+$", item) is not None


def get_tensor_index(item: bytes):
    return int(item[3:])


def pack_args_kwargs(*args, **kwargs):
    masked_flat_values, flat_tensors, tensor_to_index = [], [], {}
    for value in nested_flatten((args, kwargs)):
        if isinstance(value, torch.Tensor):
            tensor_index = tensor_to_index.setdefault(value, len(flat_tensors))
            if tensor_index == len(flat_tensors):
                flat_tensors.append(value)
            masked_flat_values.append(mark_masked_tensor(tensor_index))
        else:
            masked_flat_values.append(value)
    return flat_tensors, dict(structure=nested_pack(masked_flat_values, (args, kwargs)))


def unpack_args_kwargs(flat_tensors, metadata):
    return nested_pack(
        (
            value if not is_masked_tensor(value) else flat_tensors[get_tensor_index(value)]
            for value in nested_flatten(metadata["structure"])
        ),
        metadata["structure"],
    )
