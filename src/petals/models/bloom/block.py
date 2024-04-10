"""
Bloom intermediate layer
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""
from typing import Optional, Tuple

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomModel, build_alibi_tensor

from petals.utils.misc import is_dummy


class WrappedBloomBlock(BloomBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ):
        assert attention_mask is None, "Non-causal attention masks are not supported yet"
        batch_size, seq_length = hidden_states.shape[:2]
        if layer_past is not None and is_dummy(layer_past[0]):
            # Bloom cannot use cache if it was misconsctructed(e.g. Dummy tensors)
            # In this case, fallback to the old code:
            layer_past = None
        past_length = 0 if layer_past is None else layer_past[0].shape[-1]
        seq_length_with_past = seq_length + past_length
        attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        if alibi is None:
            alibi = build_alibi_tensor(attention_mask, num_heads=self.num_heads, dtype=hidden_states.dtype)
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=hidden_states,
            past_key_values_length=past_length,
        )
        attention_mask = attention_mask.bool()
        return super().forward(
            hidden_states, *args, attention_mask=attention_mask, alibi=alibi, layer_past=layer_past, **kwargs
        )
