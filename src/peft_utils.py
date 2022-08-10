"""

Generalized parameter-efficient finetuning modules that support deep prompts and several types of adapters.
Designed to be used on both client and server side.

"""
import torch.nn as nn

from src.utils.misc import DUMMY


class GenericPEFTModule(nn.Module):
    """Container for PEFT parameters for a single transformer block, supports multiple modes"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.prompts = nn.Parameter(DUMMY)
