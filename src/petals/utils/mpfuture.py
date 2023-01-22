import torch
from hivemind.utils import mpfuture


class DummySharedBytes:
    @classmethod
    def next(cls):
        return torch.zeros([1], dtype=torch.uint8)


def patch_mpfuture():
    """
    Shared memory inside hivemind.utils.MPFuture still leads to regular server crashes.
    However, it is only used for cancelling MPFutures, which is not used across petals for now.
    This function monkey-patches hivemind to remove this functionality.
    """

    mpfuture.SharedBytes = DummySharedBytes
