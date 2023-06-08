"""
Bloom intermediate layer
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""
import os
from typing import Optional, Tuple

import torch.nn.quantized.dynamic.modules.linear
import transformers
from packaging import version
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# if not os.getenv("PETALS_IGNORE_DEPENDENCY_VERSION"):
#     assert (
#         version.parse("4.25.1") <= version.parse(transformers.__version__) < version.parse("5.0.0")
#     ), "Please install a proper transformers version: pip install transformers>=4.25.1,<5.0.0"


class WrappedLlamaBlock(LlamaDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ):
        return super().forward(hidden_states, *args, past_key_value=layer_past, **kwargs)
