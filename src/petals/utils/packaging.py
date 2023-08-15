from typing import Any, Dict, List, Tuple

import torch
from hivemind import nested_flatten, nested_pack

# TODO: Move functions to hivemind


def _mark_masked_tensor(index: int) -> bytes:
    return b"__T" + str(index).encode()


def _is_masked_tensor(item: Any) -> bool:
    return isinstance(item, bytes) and item.startswith(b"__T")


def _get_tensor_index(item: bytes) -> int:
    return int(item[3:])


def pack_args_kwargs(*args, **kwargs) -> Tuple[List[torch.Tensor], Any]:
    """
    Check the function's arguments and pack all tensors into different flattened lists.
    :returns: a flattened list of tensors and args and kwargs, where tensors were masked
    """
    masked_flat_values, flat_tensors, tensor_to_index = [], [], {}
    for value in nested_flatten((args, kwargs)):
        if isinstance(value, torch.Tensor):
            tensor_index = tensor_to_index.setdefault(value, len(flat_tensors))
            if tensor_index == len(flat_tensors):
                flat_tensors.append(value)
            masked_flat_values.append(_mark_masked_tensor(tensor_index))
        else:
            masked_flat_values.append(value)
    return flat_tensors, nested_pack(masked_flat_values, (args, kwargs))


def unpack_args_kwargs(flat_tensors: List[torch.Tensor], args_structure: Any):
    """
    Restore arguments after `pack_args_kwargs` function.
    :returns: list of args and dict of kwargs
    """
    return nested_pack(
        (
            value if not _is_masked_tensor(value) else flat_tensors[_get_tensor_index(value)]
            for value in nested_flatten(args_structure)
        ),
        args_structure,
    )
