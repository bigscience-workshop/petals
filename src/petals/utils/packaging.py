import torch
from hivemind import nested_flatten, nested_pack

# TODO: Move functions to hivemind

MASKED_TENSOR = b'__T'


def pack_args_kwargs(*args, **kwargs):
    flat_tensors, masked_flat_values = [], []
    for value in nested_flatten((args, kwargs)):
        value_is_tensor = isinstance(value, torch.Tensor)
        masked_flat_values.append(MASKED_TENSOR if value_is_tensor else value)
        if value_is_tensor:
            flat_tensors.append(value)
    return flat_tensors, nested_pack(masked_flat_values, (args, kwargs))


def unpack_args_kwargs(flat_tensors, structure):
    tensor_iter = iter(flat_tensors)
    return nested_pack((
        value if value != MASKED_TENSOR else next(tensor_iter)
        for value in nested_flatten(structure)
    ), structure)
