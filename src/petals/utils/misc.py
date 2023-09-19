import torch

DUMMY = torch.empty(0)  # dummy tensor that replaces empty prompt or adapter parameters

DUMMY_INT64 = torch.empty(0, dtype=torch.int64)


def is_dummy(tensor: torch.Tensor) -> bool:
    return tensor.numel() == 0


SPECIAL_DTYPE_SIZES = {torch.bool: 1, torch.qint8: 1, torch.qint32: 4}


def get_size_in_bytes(dtype: torch.dtype) -> int:
    if dtype in SPECIAL_DTYPE_SIZES:
        return SPECIAL_DTYPE_SIZES[dtype]
    get_info = torch.finfo if dtype.is_floating_point else torch.iinfo
    return (get_info(dtype).bits * (1 + dtype.is_complex)) // 8


def docstring_from(source):
    def add_docstring(dest):
        dest.__doc__ = source.__doc__
        return dest

    return add_docstring
