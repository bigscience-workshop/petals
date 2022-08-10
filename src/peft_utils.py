"""
Generalized parameter-efficient finetuning module that supports deep prompts, bitfit, and several types of adapters.
Designed to be used on both client and server side.

Note: if you want to fine-tune a model in a way that is not covered by this module, please implement the
necessary parts on client side and keep the server-side code unchanged.
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
        self.output_prompts = nn.Parameter(DUMMY)   # dummy or [batch_size or 1, seq_length_prefix, hid_size]
        self.attention_query_adapter = GenericAdapter(self.hidden_size, self.hidden_size)
        self.attention_key_adapter = GenericAdapter(self.hidden_size, self.hidden_size)
        self.attention_value_adapter = GenericAdapter(self.hidden_size, self.hidden_size)
        self.attention_out_adapter = GenericAdapter(self.hidden_size, self.hidden_size)
        self.mlp_in_adapter = GenericAdapter(self.hidden_size, self.hidden_size)
        self.mlp_out_adapter = GenericAdapter(self.hidden_size, self.hidden_size)


# planned:
# strategy: define
# - check that LowRankAdapter works :)
# - implement a function that converts lowrank adapter to [list_of_tensors, metadata]
# - pass list of tensors and metadata in chained requests
# - figure out how to handle layernorm, e.g. option to normalize before adapter(default=True, no rescale)
# - check exact match with local layer


class GenericAdapter(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.in_proj = nn.Parameter(DUMMY, requires_grad=False)         # [rank, in_features]
        self.hid_bias = nn.Parameter(DUMMY, requires_grad=False)        # [rank]
        self.out_proj = nn.Parameter(DUMMY, requires_grad=False)        # [out_features, rank]
        self.out_bias = nn.Parameter(DUMMY, requires_grad=False)        # [out_features]
        self.out_scale_proj = nn.Parameter(DUMMY, requires_grad=False)  # [out_features, rank]
        self.out_scale = nn.Parameter(DUMMY, requires_grad=False)       # [out_features]
        self.register_buffer("activation", torch.tensor(0, torch.int64), persistent=True)  # []

    def forward(self, input: torch.Tensor, base_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param input: applies adapter to this tensor
        :param base_output: outputs of a base model's linear layer; defaults to same as input
        :return: adjusted output, after using the low-rank adapter
        """
        base_output = base_output if base_output is not None else input
        dtype, device = input.dtype, input.device
        has_scale, has_bias = not is_dummy(self.out_scale), not is_dummy(self.out_bias)
        has_adapter = not is_dummy(self.in_proj)

        # adapter components
        additive = self.out_bias if has_bias else torch.zeros(self.out_features, dtype=dtype, device=device)
        multiplicative = self.out_scale if has_scale else torch.ones(self.out_features, dtype=dtype, device=device)

        if has_adapter:
            hid = F.linear(input, weight=self.in_proj, bias=None if is_dummy(self.in_bias) else self.in_bias)
            hid = _ACTIVATIONS_BY_INDEX[int(self.activation.item())](hid)
            if not is_dummy(self.out_proj):
                additive = F.linear(hid, self.out_proj, bias=additive)
            if not is_dummy(self.out_scale_proj):
                multiplicative = F.linear(hid, self.out_scale_proj, bias=multiplicative)
        return torch.addcmul(additive, base_output, multiplicative)

    @property
    def rank(self) -> int:
        return 0 if is_dummy(self.out_proj) else self.out_proj.shape[-1]


class ACTIVATIONS(Enum):
    # enum of allowed activations for server-side adapters; linear activation is represented with DUMMY tensor
    # beware: these activations should be backwards compatible! new activations can only be added to the end of the list
    linear, relu, gelu, relu6, leaky_relu, sigmoid, tanh = range(7)


for act in list(ACTIVATIONS)[1:]:
    assert hasattr(F, act.name), act.name

_ACTIVATIONS_BY_INDEX = {act.value: getattr(F, act.name) for act in list(ACTIVATIONS)[1:]}
_ACTIVATIONS_BY_INDEX[0] = lambda x: x

