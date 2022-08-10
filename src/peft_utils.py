"""

Generalized parameter-efficient finetuning module that supports deep prompts, bitfit, and several types of adapters.
Designed to be used on both client and server side.

"""
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.misc import DUMMY, is_dummy


class TransformerBlockPEFT(nn.Module):
    """
    Modular parameter-efficient finetuning adapters for a single transformer block.
    Contains a variable number of parameters that can provide soft prompts, adapters, IA3, or a combination thereof.

    :note: all unused trainable parameters will be represented with a special DUMMY tensor
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # "deep" prompts, applied to the outputs of each layer (https://arxiv.org/abs/2110.07602)
        self.prompts = nn.Parameter(DUMMY)   # dummy or [batch_size or 1, seq_length_prefix, hid_size]

        # adapter input projection; used for output adapters, can be reused for other adapters
        self.key_adapter = LowRankAdapter
        self.adapter_in_bias = nn.Parameter(DUMMY)  # [hid_size]

        # output projection, applied to the residual layer after MLP
        self.adapter_out_weight = nn.Parameter(DUMMY)  # [adapter_dim, hid_size or hid_size * 2]
        self.adapter_out_bias = nn.Parameter(DUMMY)  # [hid_size]
        self.adapter_out_scale = nn.Parameter(DUMMY)  # [hid_size]

# planned:
# strategy: define
# - remove the part that stacks multiplicative and additive adapter weights - it does not help!
# - check that LowRankAdapter works :)
# - implement a function that converts lowrank adapter to [list_of_tensors, metadata]
# - pass list of tensors and metadata in chained requests
# - figure out how to handle layernorm, e.g. option to normalize before adapter(default=True, no rescale)
# - check exact match with local layer


class LowRankAdapter(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_proj = nn.Parameter(DUMMY, requires_grad=False)     # [rank, hid_size]
        self.hid_bias = nn.Parameter(DUMMY, requires_grad=False)    # [rank]
        self.out_proj = nn.Parameter(DUMMY, requires_grad=False)    # [hid_size or 2 * hid_size, rank]
        self.out_scale = nn.Parameter(DUMMY, requires_grad=False)   # [hid_size]
        self.out_bias = nn.Parameter(DUMMY, requires_grad=False)    # [hid_size]
        self.register_buffer("activation", torch.tensor(0, torch.int64), persistent=True)  # []

    def forward(self, input: torch.Tensor, base_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param input: applies adapter to this tensor
        :param base_output: outputs of a base model's linear layer; defaults to same as input
        :return: adjusted output, after using the low-rank adapter
        """
        base_output = base_output if base_output is not None else input
        has_scale, has_bias = not is_dummy(self.out_scale), not is_dummy(self.out_bias)
        has_adapter = not is_dummy(self.in_proj)

        # adapter components
        additive = self.out_bias if has_bias else None
        multiplicative = self.out_scale if has_scale else None

        if has_adapter:
            hid = F.linear(input, weight=self.in_proj, bias=None if is_dummy(self.in_bias) else self.in_bias)

            if self.activation:
                activation_fn = _ACTIVATIONS_BY_INDEX[int(self.activation.item())]
                hid = activation_fn(hid)

            if self.out_proj.shape[0] == self.hidden_size:
                additive = F.linear(hid, self.out_proj, bias=additive)

            elif self.out_proj.shape[0] == 2 * self.hidden_size:
                bias_and_scale = None
                if has_scale or has_bias:
                    scale_or_ones = self.out_scale if has_scale else torch.ones_like(self.out_bias)
                    bias_or_zeros = self.out_bias if has_bias else torch.zeros_like(self.out_scale)
                    bias_and_scale = torch.cat([bias_or_zeros, scale_or_ones], dim=0)
                combined_out = F.linear(hid, self.out_proj, bias=bias_and_scale)
                additive, multiplicative = combined_out.split(self.hidden_size, dim=-1)

        if additive is not None and multiplicative is not None:
            return torch.addcmul(additive, base_output, multiplicative)
        elif additive is not None:
            return additive.add_(base_output)
        elif multiplicative is not None:
            return base_output * multiplicative
        else:
            return base_output

    @property
    def rank(self) -> int:
        return 0 if is_dummy(self.out_proj) else self.out_proj.shape[-1]


class ACTIVATIONS(Enum):
    # enum of allowed activations for server-side adapters; linear activation is represented with DUMMY tensor
    # beware: these activations should be backwards compatible! new activations can only be added to the end of the list
    relu, gelu, relu6, leaky_relu, sigmoid, tanh = range(1, 7)


for act in list(ACTIVATIONS)[1:]:
    assert hasattr(F, act.name), act.name

_ACTIVATIONS_BY_INDEX = {act.value: getattr(F, act.name) for act in ACTIVATIONS}
_ACTIVATIONS_BY_INDEX[0] = lambda x: x

