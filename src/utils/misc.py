import torch

DUMMY = torch.empty(0)  # dummy tensor that replaces empty prompt or adapter parameters
make_dummy_batch = lambda x: torch.empty(x)


def is_dummy(tensor: torch.Tensor):
    return tensor.numel() == 0


def is_dummy_batch(tensor: torch.Tensor, batch_size: int):
    return tensor.numel() == batch_size and tensor.ndim == 1
