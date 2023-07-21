

# Graciously stolen from the legend himself, John Smith.
# https://raw.githubusercontent.com/johnsmith0031/alpaca_lora_4bit/winglian-setup_pip/src/alpaca_lora_4bit/monkeypatch/llama_flash_attn_monkey_patch.py


from typing import List, Optional, Tuple
import torch
from torch import nn
import transformers
from einops import rearrange
import math

try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
except ImportError:
    raise ImportError("Please install flash_attn to use this module")


def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = flash_attn_forward_gqa
    print("Replaced attention with flash_attn")


def flash_attn_forward_gqa(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        #We only apply flash attention optimizations if we don't need to output the whole attention matrix
        if not output_attentions:
            
            # GQA Support
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            
            # q: (batch_size, seqlen, nheads, headdim)
            # k: (batch_size, seqlen, nheads_k, headdim)
            # v: (batch_size, seqlen, nheads_k, headdim)
            # flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

            if attention_mask is None or (attention_mask.shape[3] > 1 and attention_mask[0, 0, 0, 1] == 0):
                attn_output = flash_attn_func(query_states, key_states, value_states, dropout_p=0.0, softmax_scale=None)
            else:
                attn_output = flash_attn_func(query_states, key_states, value_states, dropout_p=0.0, softmax_scale=None, causal=True)
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
