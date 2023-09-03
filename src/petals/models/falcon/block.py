"""
Falcon intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
See commit history for authorship.
"""
import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    FalconConfig,
    FalconDecoderLayer,
    FalconLinear,
    FalconMLP,
    FalconModel,
    FalconRotaryEmbedding,
    LayerNorm,
    build_alibi_tensor,
    dropout_add,
    rotate_half,
)


KVCache = Tuple[torch.Tensor, torch.Tensor]
INFERENCE_MAX_LENGTH = 8192


def apply_rotary(query, key, cos, sin):
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


class OptimizedFalconRotaryEmbedding(FalconRotaryEmbedding):
    def __init__(self, head_dim: int, base=10000):
        super().__init__(head_dim, base)
        self.cuda_graph = None
        self.input_surface = None
        self.static_outputs = None

    def _optimized_apply_rotary(self, query, key, cos, sin):
        if self.cuda_graph is None:
            self.cuda_graph = torch.cuda.CUDAGraph()
            self.input_surface = (query, key, cos, sin)

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    apply_rotary(*self.input_surface)
            torch.cuda.current_stream().wait_stream(s)

            with torch.cuda.graph(self.cuda_graph):
                self.static_outputs = apply_rotary(*self.input_surface)

        inputs = (query, key, cos, sin)
        for static_input, data in zip(self.input_surface, inputs):
            static_input.copy_(data)
        self.cuda_graph.replay()
        return tuple(o.detach() for o in self.static_outputs)

    def cos_sin(self, seq_len: int, past_key_values_length: int, device="cpu", dtype=torch.bfloat16) -> torch.Tensor:
        if self.seq_len_cached == -1:
            # warm up the cache
            super().cos_sin(1, INFERENCE_MAX_LENGTH - 1, device=device, dtype=dtype)
        return super().cos_sin(
            seq_len=seq_len, past_key_values_length=past_key_values_length, device=device, dtype=dtype
        )

    def forward(self, query, key, past_key_values_length=0):
        batch, seq_len, head_dim = query.shape
        cos, sin = self.cos_sin(seq_len, past_key_values_length, query.device, query.dtype)
        if seq_len == 1 and torch.is_inference_mode_enabled():
            return self._optimized_apply_rotary(query, key, cos, sin)
        else:
            return apply_rotary(query, key, cos, sin)


def split_heads(
    fused_qkv: torch.Tensor, num_heads, num_kv_heads, head_dim
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, _ = fused_qkv.shape
    qkv = fused_qkv.view(batch, seq_len, -1, num_heads // num_kv_heads + 2, head_dim)
    query, key, value = torch.split(qkv, [num_heads // num_kv_heads, 1, 1], dim=3)
    key = torch.broadcast_to(key, query.shape)
    value = torch.broadcast_to(value, query.shape)

    query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
    return query, key, value


class OptimizedFalconAttention(FalconAttention):
    def __init__(self, config: FalconConfig):
        nn.Module.__init__(self)
        assert config.new_decoder_architecture
        assert config.attention_dropout == 0.0

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.maybe_rotary = OptimizedFalconRotaryEmbedding(config.head_dim) if config.rotary else lambda q, k, t: (q, k)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor

        qkv_out_dim = (config.num_kv_heads * 2 + config.num_attention_heads) * self.head_dim

        self.query_key_value = FalconLinear(self.hidden_size, qkv_out_dim, bias=config.bias)
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.dense = FalconLinear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.num_kv_heads = config.num_kv_heads

        self._split_heads = partial(
            split_heads, num_heads=self.num_heads, num_kv_heads=self.num_kv_heads, head_dim=self.head_dim
        )
        self.qkv_graph = None
        self.input_surface = None
        self.static_outputs = None

    def _optimized_apply_qkv(self, hidden_states):
        if self.qkv_graph is None:
            self.qkv_graph = torch.cuda.CUDAGraph()
            self.static_input = hidden_states

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fused_qkv = self.query_key_value(hidden_states)
                    self._split_heads(fused_qkv)
            torch.cuda.current_stream().wait_stream(s)

            with torch.cuda.graph(self.qkv_graph):
                static_fused_qkv = self.query_key_value(hidden_states)
                self.static_outputs = self._split_heads(static_fused_qkv)

        self.static_input.copy_(hidden_states)
        self.qkv_graph.replay()
        return tuple(o.detach() for o in self.static_outputs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        assert alibi is None
        assert not output_attentions

        if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled():
            query_layer, key_layer, value_layer = self._optimized_apply_qkv(hidden_states)
        else:
            fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
            # 3 x [batch_size, seq_length, num_heads, head_dim]
            (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        num_kv_heads = self.num_heads
        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(
            batch_size * num_kv_heads,
            query_length,
            self.head_dim,
        )
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)

        past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
        query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, past_kv_length)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, kv_length, head_dim]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)

        attn_output = F.scaled_dot_product_attention(
            query_layer_, key_layer_, value_layer_, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

        output_tensor = self.dense(attn_output)

        return output_tensor, present


class OptimizedFalconDecoderLayer(FalconDecoderLayer):
    def __init__(self, config: FalconConfig):
        nn.Module.__init__(self)
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.self_attention = OptimizedFalconAttention(config)
        self.mlp = FalconMLP(config)
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        assert config.new_decoder_architecture
        self.ln_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.ln_graph = None
        self.static_input = None

    def _optimized_apply_ln(self, hidden_states):
        if self.ln_graph is None:
            self.ln_graph = torch.cuda.CUDAGraph()
            self.static_input = hidden_states

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.ln_attn(hidden_states)
                    self.ln_mlp(hidden_states)
            torch.cuda.current_stream().wait_stream(s)

            with torch.cuda.graph(self.ln_graph):
                ln_attn_output = self.ln_attn(hidden_states)
                ln_mlp_output = self.ln_mlp(hidden_states)
                self.static_outputs = (ln_attn_output, ln_mlp_output)

        self.static_input.copy_(hidden_states)
        self.ln_graph.replay()
        return tuple(o.detach() for o in self.static_outputs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states

        if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled():
            attention_layernorm_out, mlp_layernorm_out = self._optimized_apply_ln(hidden_states)
        else:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)

        attn_outputs = self.self_attention(
            attention_layernorm_out,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        mlp_output = self.mlp(mlp_layernorm_out)
        mlp_output += attention_output

        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class _WrappedFalconBlock(OptimizedFalconDecoderLayer):
    def __init__(self, config: FalconConfig):
        super().__init__(config)
        assert not self.config.alibi

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        assert attention_mask is None

        if layer_past is not None:
            layer_past = self._reorder_cache_from_bloom_to_falcon(layer_past)

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=None,
            alibi=None,
            layer_past=layer_past,
            use_cache=use_cache,
            **kwargs,
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

        state = state.view(batch_size, 1, self.config.num_kv_heads, seq_len, head_dim)
        # Here, .expand() doesn't allocate new memory, instead uses stride=0 along dim=1
        state = state.expand(-1, self.config.num_key_value_groups, -1, -1, -1)
        state = state.reshape(batch_size * self.config.num_attention_heads, seq_len, head_dim)
        return state

    def _collapse_states(self, state: torch.Tensor) -> torch.Tensor:
        batch_size_x_num_attn_heads, seq_len, head_dim = state.shape
        batch_size = batch_size_x_num_attn_heads // self.config.num_attention_heads

        state = state.view(batch_size, self.config.num_key_value_groups, self.config.num_kv_heads, seq_len, head_dim)
        state = state[:, 0]
        state = state.view(batch_size * self.config.num_kv_heads, seq_len, head_dim)
        return state


class WrappedFalconBlock(FalconDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[KVCache] = None,
        layer_past: Optional[KVCache] = None,
        use_cache: bool = False,
        **kwargs,
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
            **kwargs,
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
