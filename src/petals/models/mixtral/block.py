import json
from typing import Optional, Tuple

import torch
from transformers import MixtralConfig
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralModel


class WrappedMixtralBlock(MixtralDecoderLayer):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self._attn_implementation = config._attn_implementation
        self.sliding_window = config.sliding_window
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        batch_size, seq_length, _ = hidden_states.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        past_key_value = layer_past

        if past_key_value is not None:
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            _past_key_value = self._reorder_cache_from_bloom(past_key_value, batch_size, past_key_values_length)
            past_key_value = DynamicCache()
            past_key_value.key_cache = [torch.empty(0) for _ in range(self.layer_idx)] + [_past_key_value[0]]
            past_key_value.value_cache = [torch.empty(0) for _ in range(self.layer_idx)] + [_past_key_value[1]]
            past_key_value._seen_tokens = past_key_values_length

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa":
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
                sliding_window=self.sliding_window,
            )

        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=hidden_states.device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs
        )

        if use_cache:
            present_key_value = outputs[-1]
            present_key_value = present_key_value[self.layer_idx]
            present_key_value = self._reorder_cache_to_bloom(present_key_value, batch_size, seq_length_with_past)
            outputs = outputs[:-1] + (present_key_value,)

        return outputs

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        # TODO: Move to mixin
        key_states, value_states = key_value
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.view(
            batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        value_states = value_states.view(*key_states.shape)
        return (key_states, value_states)

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        # TODO: Move to mixin
        key_states, value_states = key_value
        value_states = value_states.view(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.view(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)
