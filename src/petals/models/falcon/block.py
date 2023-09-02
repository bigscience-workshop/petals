"""
Falcon intermediate layer
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""
from typing import Optional, Tuple

import torch
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer, FalconModel, build_alibi_tensor


class WrappedFalconBlock(FalconDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        if layer_past is not None:
            layer_past = self._reorder_cache_from_bloom_to_falcon(layer_past)
        past_length = 0 if layer_past is None else layer_past[0].shape[1]
        seq_length_with_past = seq_length + past_length

        attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        if alibi is None and self.config.alibi:
            alibi = build_alibi_tensor(attention_mask, num_heads=self.num_heads, dtype=hidden_states.dtype)
        attention_mask = FalconModel._prepare_attn_mask(attention_mask, (batch_size, seq_length), past_length)

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            alibi=alibi,
            layer_past=layer_past,
            use_cache=use_cache,
            **kwargs
        )

        if use_cache:
            present_key_value = outputs[-1]
            present_key_value = self._reorder_cache_from_bloom_to_falcon(present_key_value)
            outputs = outputs[:-1] + (present_key_value,)

        return outputs

    @staticmethod
    def _reorder_cache_from_bloom_to_falcon(
        key_value: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)
