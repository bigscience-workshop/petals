import torch

DUMMY = torch.empty(0)  # dummy tensor that replaces empty prompt or adapter parameters

DUMMY_INT64 = torch.empty(0, dtype=torch.int64)


def is_dummy(tensor: torch.Tensor):
    return tensor.numel() == 0
