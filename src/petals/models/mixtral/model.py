from typing import Optional

import torch
import torch.nn as nn
from hivemind import DHT
from hivemind.utils.logging import get_logger
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.mixtral import (
    MixtralForCausalLM,
    MixtralForSequenceClassification,
    MixtralModel,
    MixtralPreTrainedModel,
)

from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.lm_head import LMHead
from petals.client.ptune import PTuneMixin
from petals.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from petals.client.remote_sequential import RemoteSequential
from petals.models.mixtral.config import DistributedMixtralConfig
from petals.utils.auto_config import DefaultRevisionMixin

logger = get_logger(__name__)


class DistributedMixtralModel(DefaultRevisionMixin, FromPretrainedMixin, PTuneMixin, MixtralModel):
    """MixtralModel, but all transformer layers are hosted by the swarm"""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedMixtralConfig

    def __init__(self, config: DistributedMixtralConfig, *, dht: Optional[DHT] = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0  # Prevent initialization
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer

        self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)  # Forbid accumulate grads for embeddings and layernorm
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # The causal mask will be added on the server-side
        assert (
            attention_mask is None or (attention_mask == 1).all()
        ), f"Custom attention masks are not supported, {attention_mask=}"
        assert (
            position_ids is None or (position_ids[:, 1:] - position_ids[:, :-1] == 1).all()
        ), f"Non-consecutive position_ids are not supported, {position_ids=}"
        assert head_mask is None, f"Custom head masks are not supported, {head_mask=}"
        assert use_cache is None or use_cache, f"{use_cache=} is not supported"
        assert not output_attentions, f"{output_attentions=} is not supported"
        assert not output_hidden_states, f"{output_hidden_states=} is not supported"
        assert return_dict is None or return_dict, f"{return_dict=} is not supported"
        assert not output_router_logits, f"{output_router_logits=} is not supported"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_prompts = self.config.tuning_mode and "ptune" in self.config.tuning_mode and self.h.position == 0
        if use_prompts:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            prompts = intermediate_prompts = None

        hidden_states = inputs_embeds
        output_shape = input_shape + (hidden_states.size(-1),)

        if past_key_values is None:
            past_key_values = RemotePastKeyValues()
        past_key_values.update_seen(hidden_states.size(1))

        hidden_states = self.layers(
            hidden_states,
            prompts=intermediate_prompts,
            hypo_ids=past_key_values.hypo_ids if past_key_values is not None else None,
        )

        # Remove prefix
        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        # Add last hidden state
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    @property
    def word_embeddings(self) -> nn.Embedding:  # For compatibility with RemoteGenerationMixin
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin in tests
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:  # For compatibility with RemoteGenerationMixin
        return self.layers

    @property
    def ln_f(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin in tests
        return self.norm


class DistributedMixtralForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, MixtralForCausalLM):
    _keys_to_ignore_on_load_missing = DistributedMixtralModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedMixtralModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedMixtralConfig

    def __init__(self, config: DistributedMixtralConfig):
        MixtralPreTrainedModel.__init__(self, config)
        self.model = DistributedMixtralModel(config)
        self.lm_head = LMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedMixtralModel:  # For compatibility with RemoteGenerationMixin
        return self.model


class DistributedMixtralForSequenceClassification(FromPretrainedMixin, MixtralForSequenceClassification):
    _keys_to_ignore_on_load_missing = DistributedMixtralModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedMixtralModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedMixtralConfig

    def __init__(self, config: DistributedMixtralConfig):
        MixtralPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        self.model = DistributedMixtralModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def transformer(self) -> DistributedMixtralModel:  # For compatibility with RemoteGenerationMixin
        return self.model
