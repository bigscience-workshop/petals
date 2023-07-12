from enum import Enum

import torch


class QuantType(Enum):
    NONE = 0
    INT8 = 1  # 8-bit as in the LLM.int8() paper
    NF4 = 2  # 4-bit as in the QLoRA paper


DUMMY = torch.empty(0)  # dummy tensor that replaces empty prompt or adapter parameters


def is_dummy(tensor: torch.Tensor):
    return tensor.numel() == 0
