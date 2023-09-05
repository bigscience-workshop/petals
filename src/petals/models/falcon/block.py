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
    LayerNorm,
    build_alibi_tensor,
    dropout_add,
    rotate_half,
)

KVCache = Tuple[torch.Tensor, torch.Tensor]
INFERENCE_MAX_LENGTH = 8192


def apply_rotary(query, key, cos, sin):
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


class OptimizedFalconRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = -1

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
        total_length = seq_len + past_key_values_length
        if self.seq_len_cached == -1:
            # warm up the cache
            total_length = max(INFERENCE_MAX_LENGTH, total_length)

        if total_length > self.seq_len_cached:
            with torch.inference_mode(False):
                self.seq_len_cached = total_length
                t = torch.arange(total_length, device=device, dtype=self.inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, self.inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1).to(device)

                if dtype in [torch.float16, torch.bfloat16]:
                    emb = emb.float()

                self.register_buffer("cos_cached", emb.cos()[None, :, :].type(dtype), persistent=False)
                self.register_buffer("sin_cached", emb.sin()[None, :, :].type(dtype), persistent=False)

        return (
            self.cos_cached[:, past_key_values_length : seq_len + past_key_values_length].type(dtype),
            self.sin_cached[:, past_key_values_length : seq_len + past_key_values_length].type(dtype),
        )

    def forward(self, query, key, past_key_values_length=0):
        batch, seq_len, head_dim = query.shape
        cos, sin = self.cos_sin(seq_len, past_key_values_length, query.device, query.dtype)
        if seq_len == 1 and torch.is_inference_mode_enabled() and query.device.type == "cuda":
            return self._optimized_apply_rotary(query, key, cos, sin)
        else:
            return apply_rotary(query, key, cos, sin)


def split_heads(
    fused_qkv: torch.Tensor, num_heads: int, num_kv_heads: int, head_dim: int
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
        if config.new_decoder_architecture:
            qkv_out_dim = (config.num_kv_heads * 2 + config.num_attention_heads) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        self.query_key_value = FalconLinear(self.hidden_size, qkv_out_dim, bias=config.bias)
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.dense = FalconLinear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv_heads = config.num_kv_heads if (self.new_decoder_architecture or not self.multi_query) else 1

        if self.new_decoder_architecture:
            self._split_heads = partial(
                split_heads, num_heads=self.num_heads, num_kv_heads=self.num_kv_heads, head_dim=self.head_dim
            )
            self.split_graph = None
            self.input_surface = None
            self.static_outputs = None

    def _optimized_split_heads(self, fused_qkv):
        if self.split_graph is None:
            self.split_graph = torch.cuda.CUDAGraph()
            self.input_surface = fused_qkv

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._split_heads(fused_qkv)
            torch.cuda.current_stream().wait_stream(s)

            with torch.cuda.graph(self.split_graph):
                self.static_outputs = self._split_heads(self.input_surface)

        self.input_surface.copy_(fused_qkv)
        self.split_graph.replay()
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
        assert not output_attentions

        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        if (
            self.new_decoder_architecture
            and hidden_states.size(1) == 1
            and torch.is_inference_mode_enabled()
            and hidden_states.device.type == "cuda"
        ):
            query_layer, key_layer, value_layer = self._optimized_split_heads(fused_qkv)
        else:
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

        _, kv_length, _ = key_layer.shape
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)

        attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, float("-1e9")).to(query_layer.dtype)

        if alibi is None:
            attn_output = F.scaled_dot_product_attention(
                query_layer_, key_layer_, value_layer_, attn_mask=attention_mask_float, dropout_p=0.0, is_causal=False
            )

            attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

            output_tensor = self.dense(attn_output)

            return output_tensor, present
        else:
            matmul_result = query_layer_ @ key_layer_.transpose(-1, -2)

            # change view to [batch_size, num_heads, q_length, kv_length]
            attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)

            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            input_dtype = attention_scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                attention_scores = attention_scores.to(torch.float32)
            # Matt (HF) note: We could possibly use F.scaled_dot_product_attention here too, by
            # adding (alibi * self.inv_norm_factor) to attention_mask_float. I think this would be mathematically
            # equivalent and more performant, but there might be a numerical difference. If you're reading this
            # and you'd like to experiment and maybe file a PR, feel free!
            attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
            attention_logits *= self.inv_norm_factor
            attention_probs = F.softmax(attention_logits + attention_mask_float, dim=-1, dtype=hidden_states.dtype)
            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attention_dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # change view [batch_size, num_heads, q_length, kv_length]
            attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)

            # matmul: [batch_size * num_heads, q_length, head_dim]
            context_layer = (attention_probs_reshaped @ value_layer_).flatten(0, 1)

            # change view [batch_size, q_length, num_heads * head_dim]
            context_layer = self._merge_heads(context_layer)

            output_tensor = self.dense(context_layer)

            if output_attentions:
                return output_tensor, present, attention_probs
            else:
                return output_tensor, present


class OptimizedFalconDecoderLayer(FalconDecoderLayer):
    def __init__(self, config: FalconConfig):
        nn.Module.__init__(self)
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.mlp = FalconMLP(config)
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        self.self_attention = OptimizedFalconAttention(config)

        if self.config.alibi or not config.new_decoder_architecture:
            if config.new_decoder_architecture:
                # The layer norm before self-attention
                self.ln_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
                # The layer norm before the MLP
                self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            else:
                self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
                if not config.parallel_attn:
                    self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        else:
            self.ln_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

            self.ln_graph = None
            self.static_input = None
            self.static_outputs = None

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

        if self.config.new_decoder_architecture:
            if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
                attention_layernorm_out, mlp_layernorm_out = self._optimized_apply_ln(hidden_states)
            else:
                attention_layernorm_out = self.ln_attn(hidden_states)
                mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

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

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    attention_output, residual, self.config.attention_dropout, training=self.training
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]

        mlp_output = self.mlp(mlp_layernorm_out)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class WrappedFalconBlock(OptimizedFalconDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[KVCache] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        assert attention_mask is None

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
