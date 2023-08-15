import torch

DUMMY = torch.empty(0)  # dummy tensor that replaces empty prompt or adapter parameters

DUMMY_INT64 = torch.empty(0, dtype=torch.int64)


def is_dummy(tensor: torch.Tensor) -> bool:
    return tensor.numel() == 0


def docstring_from(source):
    def add_docstring(dest):
        dest.__doc__ = source.__doc__
        return dest

    return add_docstring
