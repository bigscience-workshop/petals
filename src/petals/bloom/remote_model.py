from typing import Optional

import hivemind
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bloom import (
    BloomConfig,
    BloomForCausalLM,
    BloomForSequenceClassification,
    BloomModel,
    BloomPreTrainedModel,
)

from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.modeling_utils import LMHead, force_non_empty_weights
from petals.client.remote_generation import RemoteGenerationMixin
from petals.client.remote_model import DistributedPretrainedConfig
from petals.client.remote_sequential import RemoteSequential
from petals.utils.misc import DUMMY

logger = get_logger(__name__)


class DistributedBloomConfig(BloomConfig, DistributedPretrainedConfig):
    pass


class DistributedBloomModel(FromPretrainedMixin, BloomModel):
    """BloomModel, but all transformer layers are hosted by the swarm"""

    _keys_to_ignore_on_load_missing = BloomModel._keys_to_ignore_on_load_missing + [
        r"^(intermediate_)?prompt_embeddings\.weight$",
    ]
    _keys_to_ignore_on_load_unexpected = [r"^h\."]

    config_class = DistributedBloomConfig

    def __init__(self, config: DistributedBloomConfig, *, dht: Optional[hivemind.DHT] = None):
        assert config.dht_prefix, "Could not find dht_prefix in config, please create model with dht_prefix=..."
        assert config.initial_peers or dht is not None, "Please specify `config.initial_peers` or `dht`"

        n_layer, config.n_layer = config.n_layer, 0  # temporarily set n_layer to 0 to prevent layer initialization
        super().__init__(config)
        assert len(self.h) == 0
        config.n_layer = n_layer

        self.h = RemoteSequential(config, dht=dht)

        # Forbid accumulate grads for embeddings and layernorm
        self.set_requires_grad(False)

        if config.tuning_mode and "ptune" in config.tuning_mode:
            assert config.pre_seq_len > 0, "The number of prefix tokens must be > 0"
            self.pre_seq_len = config.pre_seq_len
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()

            with force_non_empty_weights():
                if self.word_embeddings_layernorm.weight.dtype in (torch.float16, torch.bfloat16):
                    logger.info(
                        "Prompt embeddings and their optimizer statistics will be kept in float32 "
                        "to increase ptune quality"
                    )
                self.prompt_embeddings = nn.Embedding(self.pre_seq_len, config.hidden_size, dtype=torch.float32)
                if config.tuning_mode == "deep_ptune":
                    self.intermediate_prompt_embeddings = nn.Embedding(
                        self.pre_seq_len,
                        config.num_hidden_layers * config.hidden_size,
                        # ^-- TODO: should be num_hidden_layers - 1
                        dtype=torch.float32,
                    )
        elif config.tuning_mode:
            raise NotImplementedError(f"{self.tuning_mode} mode is not supported for now")

    def set_requires_grad(self, value):
        for p in self.parameters():
            p.requires_grad = value

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        prefix_tokens = prefix_tokens.to(self.word_embeddings.weight.device)
        prompts = self.prompt_embeddings(prefix_tokens)

        if self.config.tuning_mode == "deep_ptune":
            intermediate_prompts = self.intermediate_prompt_embeddings(prefix_tokens)
            intermediate_prompts = intermediate_prompts.view(
                batch_size, self.pre_seq_len, len(self.h), self.config.hidden_size  # TODO: should be len(self.h) - 1
            )
            intermediate_prompts = intermediate_prompts.permute([2, 0, 1, 3])
        else:
            intermediate_prompts = DUMMY

        dtype = self.word_embeddings.weight.dtype
        return prompts.to(dtype), intermediate_prompts.to(dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        assert attention_mask is None, "DistributedBloomModel does not support attention masks right now"

        for k, v in kwargs.items():
            if not (v is None or v is False):
                logger.debug(f"Extra keyword arguments are not yet supported (got {k} = {v})")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.config.tuning_mode and "ptune" in self.config.tuning_mode:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        output_shape = input_shape + (hidden_states.size(-1),)

        if self.config.tuning_mode and "ptune" in self.config.tuning_mode:
            hidden_states = self.h(hidden_states, prompts=intermediate_prompts)
        else:
            hidden_states = self.h(hidden_states)

        # Remove prefix
        if self.config.tuning_mode and "ptune" in self.config.tuning_mode:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class DistributedBloomForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, BloomForCausalLM):
    """DistributedBloomForCausalLM, but all transformer layers are hosted by the swarm"""

    _keys_to_ignore_on_load_missing = (
        BloomForCausalLM._keys_to_ignore_on_load_missing
        + DistributedBloomModel._keys_to_ignore_on_load_missing
        + [r"^lm_head.word_embeddings\.weight$"]  # Missing since they are shared with input embeddings
    )
    _keys_to_ignore_on_load_unexpected = DistributedBloomModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedBloomConfig

    def __init__(self, config: DistributedBloomConfig):
        BloomPreTrainedModel.__init__(self, config)
        self.transformer = DistributedBloomModel(config)
        self.lm_head = LMHead(config, self.transformer.word_embeddings)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.word_embeddings

    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return None
        return self.lm_head

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        assert isinstance(new_embeddings, nn.Embedding)
        self.transformer.word_embeddings = self.lm_head.word_embeddings = new_embeddings
        assert self.lm_head.bias is None or len(self.lm_head.bias) == new_embeddings.num_embeddings

    def set_output_embeddings(self, new_lm_head: nn.Linear):
        with torch.no_grad():
            self.lm_head.word_embeddings.weight[...] = new_lm_head.weight
            self.lm_head.bias[...] = new_lm_head.bias


class DistributedBloomForSequenceClassification(FromPretrainedMixin, BloomForSequenceClassification):
    _keys_to_ignore_on_load_missing = (
        BloomForSequenceClassification._keys_to_ignore_on_load_missing
        + DistributedBloomModel._keys_to_ignore_on_load_missing
    )
    _keys_to_ignore_on_load_unexpected = DistributedBloomModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedBloomConfig

    def __init__(self, config: DistributedBloomConfig):
        BloomPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        self.transformer = DistributedBloomModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False).to(config.torch_dtype)

        # Initialize weights and apply final processing
        self.post_init()
