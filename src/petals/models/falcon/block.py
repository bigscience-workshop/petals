"""
Falcon intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
See commit history for authorship.
"""
from typing import Optional, Tuple

import torch
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer, FalconModel, build_alibi_tensor

KVCache = Tuple[torch.Tensor, torch.Tensor]


class WrappedFalconBlock(FalconDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[KVCache] = None,
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
            present_key_value = self._reorder_cache_from_falcon_to_bloom(present_key_value)
            outputs = outputs[:-1] + (present_key_value,)

        return outputs

    def _reorder_cache_from_bloom_to_falcon(self, key_value: KVCache) -> KVCache:
        key_states, value_states = key_value

        key_states = key_states.permute(0, 2, 1)
        assert key_states.shape == value_states.shape  # Both are [batch_size * num_kv_heads, seq_len, head_dim]

        if self.config.new_decoder_architecture:
            key_states = self._expand_states(key_states)
            value_states = self._expand_states(value_states)

        return (key_states, value_states)

    def _reorder_cache_from_falcon_to_bloom(self, key_value: KVCache) -> KVCache:
        key_states, value_states = key_value

        if self.config.new_decoder_architecture:
            key_states = self._collapse_states(key_states)
            value_states = self._collapse_states(value_states)

        assert key_states.shape == value_states.shape  # Both are [batch_size * num_kv_heads, seq_len, head_dim]
        key_states = key_states.permute(0, 2, 1)

        return (key_states, value_states)

    def _expand_states(self, state: torch.Tensor) -> torch.Tensor:
        batch_size_x_num_kv_heads, seq_len, head_dim = state.shape
        batch_size = batch_size_x_num_kv_heads // self.config.num_kv_heads

        state = state.view(batch_size, self.config.num_kv_heads, 1, seq_len, head_dim)
        state = state.expand(-1, -1, self.config.num_key_value_groups, -1, -1)  # No copy
        state = state.reshape(batch_size * self.config.num_attention_heads, seq_len, head_dim)  # Involves a copy
        return state

    def _collapse_states(self, state: torch.Tensor) -> torch.Tensor:
        batch_size_x_num_attn_heads, seq_len, head_dim = state.shape
        batch_size = batch_size_x_num_attn_heads // self.config.num_attention_heads

        state = state.view(batch_size, self.config.num_kv_heads, self.config.num_key_value_groups, seq_len, head_dim)
        state = state[:, :, 0]
        state = state.view(batch_size * self.config.num_kv_heads, seq_len, head_dim)
        return state
