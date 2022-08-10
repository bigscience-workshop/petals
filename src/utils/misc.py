import torch

DUMMY = torch.empty(0)  # dummy tensor that replaces empty prompt or adapter parameters


def is_dummy(tensor: torch.Tensor):
    return tensor.numel() == 0
