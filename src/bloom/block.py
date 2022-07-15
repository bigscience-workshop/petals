"""
Bloom intermediate layer
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""
import math

import torch
import torch.nn as nn
import torch.nn.quantized.dynamic.modules.linear

from src.bloom.ops import (
    BloomGelu,
    BloomScaledSoftmax,
    attention_mask_func,
    build_alibi_tensor,
    dropout_add,
    pre_process_alibi_for_pad,
    split_tensor_along_last_dim,
)


class BloomAttention(nn.Module):
    def __init__(self, config, layer_number=None):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.masked_softmax_fusion = config.masked_softmax_fusion
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.layer_number = max(1, layer_number)
        self.norm_factor = math.sqrt(self.head_dim) * self.layer_number

        # Scaled Softmax
        self.scale_mask_softmax = BloomScaledSoftmax(
            self.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            self.layer_number,
        )

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states,
        residual,
        layer_past=None,
        attention_mask=None,
        alibi=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if alibi is None:
            current_sequence_length = hidden_states.shape[1] + (0 if layer_past is None else layer_past[0].shape[1])
            alibi = build_alibi_tensor(
                current_sequence_length, n_head=self.num_heads, dtype=hidden_states.dtype, device=hidden_states.device
            )

        # hidden_states: [batch_size, seq_length, hidden_size]
        # apply preprocessing if the input is padded
        if attention_mask is not None:
            alibi = pre_process_alibi_for_pad(alibi, attention_mask)
        # otherwise repeat alibi tensor with the batch size
        else:
            alibi = alibi.repeat(hidden_states.shape[0], 1, 1)

        mixed_x_layer = self.query_key_value(hidden_states)

        # [batch_size, seq_length, 3 x hidden_size] --> [batch_size, seq_length, num_heads, 3 x head_dim]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_heads, 3 * self.head_dim)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [batch_size, seq_length, num_heads, 3 x head_dim] --> 3  [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=1)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=1)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [batch_size, head_dim, q_length, k_length]
        output_size = (query_layer.size(0), query_layer.size(2), query_layer.size(1), key_layer.size(1))

        # [batch_size, q_length, num_heads, head_dim] -> [q_length, batch_size * num_heads, head_dim]
        query_layer = query_layer.transpose(1, 0).reshape(output_size[2], output_size[0] * output_size[1], -1)

        # [batch_size, k_length, num_heads, head_dim] -> [k_length, batch_size * num_heads, head_dim]
        key_layer = key_layer.transpose(1, 0).reshape(output_size[3], output_size[0] * output_size[1], -1)

        # Raw attention scores. [batch_size * num_heads, q_length, k_length]
        beta = 1.0 / self.layer_number

        matmul_result = torch.baddbmm(
            alibi,
            query_layer.transpose(1, 0),
            key_layer.transpose(1, 0).transpose(1, 2),
            beta=beta,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [batch_size, num_heads, q_length, k_length]
        attention_scores = matmul_result.view(*output_size)

        # attention scores and attention mask [b, np, sq, sk]
        max_positions = max(attention_scores.shape[-1], attention_scores.shape[-2])
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask, max_positions).to(value_layer.dtype)
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context layer shape: [batch_size, num_heads, q_length, head_dim]
        output_size = (value_layer.size(0), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [k_length, batch_size x num_heads, head_dim]
        value_layer = value_layer.transpose(1, 0).reshape(value_layer.size(1), output_size[0] * output_size[1], -1)

        # change view [batch_size x num_heads, q_length, k_length]
        attention_probs_reshaped = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer.transpose(0, 1))

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = context_layer.view(*output_size)

        # [batchs_size, num_heads, q_length, head_dim] --> [q_length, batch_size, num_heads, head_dim]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [q_length, batch_size, num_heads, head_dim] --> [q_length, batch_size, hidden_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)

        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [q_length, batch_size, hidden_size]

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        output_tensor = self.dense(context_layer)
        output = output_tensor.transpose(1, 0)

        output = dropout_add(output, residual, self.hidden_dropout, self.training)

        outputs = (output, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class BloomMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dense_h_to_4h = nn.Linear(self.hidden_size, 4 * self.hidden_size)
        self.dense_4h_to_h = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.hidden_dropout = config.hidden_dropout
        self.gelu_impl = BloomGelu()

    def forward(self, hidden_states, residual):
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))
        intermediate_output = self.dense_4h_to_h(hidden_states)
        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)
        return output


class BloomBlock(nn.Module):
    def __init__(self, config, layer_number=None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.n_head = config.n_head
        self.self_attention = BloomAttention(config, layer_number=layer_number)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        alibi=None,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions
