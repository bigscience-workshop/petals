from typing import Optional, Tuple

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer, FalconModel, build_alibi_tensor
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel

from petals.server.block_utils import get_model_block
from petals.utils.auto_config import AutoDistributedConfig
from petals.utils.convert_block import QuantType, convert_block
from test_utils import MODEL_NAME

KVCache = Tuple[torch.Tensor, torch.Tensor]


class UnoptimizedWrappedFalconBlock(FalconDecoderLayer):
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


class UnoptimizedWrappedLlamaBlock(LlamaDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        batch_size, seq_length, _ = hidden_states.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        past_key_value = layer_past
        if past_key_value is not None:
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            past_key_value = self._reorder_cache_from_bloom_to_llama(past_key_value, batch_size, past_key_values_length)
        elif use_cache:
            past_key_value = DynamicCache()

        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )

        if use_cache:
            present_key_value = outputs[-1]
            present_key_value = self._reorder_cache_from_llama_to_bloom(
                present_key_value, batch_size, seq_length_with_past
            )
            outputs = outputs[:-1] + (present_key_value,)

        return outputs

    def _reorder_cache_from_bloom_to_llama(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> DynamicCache:
        key_states, value_states = key_value
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.view(
            batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        value_states = value_states.view(*key_states.shape)
        past_key_values = ((key_states, value_states),)
        return DynamicCache.from_legacy_cache(past_key_values)

    def _reorder_cache_from_llama_to_bloom(
        self, key_value: DynamicCache, batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value.to_legacy_cache()[0]
        value_states = value_states.view(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.view(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.forked
def test_optimized_block(device):
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA tests can be run only in CUDA-enabled setups")

    config = AutoDistributedConfig.from_pretrained(MODEL_NAME)

    tensor_parallel_devices = (device,)
    dtype = torch.bfloat16
    quant_type = QuantType.NONE

    block_idx = 1
    block = get_model_block(config, layer_idx=block_idx).to(dtype)
    block = convert_block(block, block_idx, config, tensor_parallel_devices, device, quant_type=quant_type, freeze=True)

    if config.model_type == "falcon":
        unopt_block = UnoptimizedWrappedFalconBlock(config).to(dtype)
    elif config.model_type == "llama":
        unopt_block = UnoptimizedWrappedLlamaBlock(config, layer_idx=0).to(dtype)
    else:
        pytest.skip(f"This test is not applicable to {config.model_type} models")

    unopt_block = convert_block(
        unopt_block, block_idx, config, tensor_parallel_devices, device, quant_type=quant_type, freeze=True
    )

    unopt_block.load_state_dict(block.state_dict())
    cache = unopt_cache = None

    with torch.inference_mode():
        for length in [10, 1, 1, 1]:
            dummy_input = torch.randn(1, length, config.hidden_size, device=device, dtype=dtype)
            block_output, cache = block(dummy_input, layer_past=cache, use_cache=True)
            unopt_block_output, unopt_cache = unopt_block(dummy_input, layer_past=unopt_cache, use_cache=True)
            assert torch.allclose(block_output, unopt_block_output, atol=1e-6, rtol=0), length
            assert torch.allclose(cache[0], unopt_cache[0], atol=1e-6, rtol=0), length
            assert torch.allclose(cache[1], unopt_cache[1], atol=1e-6, rtol=0), length
