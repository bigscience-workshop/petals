from typing import List, Optional, Tuple, Union

import hivemind
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaPreTrainedModel

from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.inference_session import InferenceSession
from petals.client.lm_head import LMHead
from petals.client.ptune import PTuneMixin
from petals.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from petals.client.remote_sequential import RemoteSequential
from petals.models.llama.config import DistributedLlamaConfig

logger = get_logger(__name__)


class DistributedLlamaModel(FromPretrainedMixin, PTuneMixin, LlamaModel):
    """LlamaModel, but all transformer layers are hosted by the swarm"""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedLlamaConfig

    def __init__(self, config: DistributedLlamaConfig, *, dht: Optional[hivemind.DHT] = None):
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
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        session: Optional[InferenceSession] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        # FIXME: Assert that the mask is None or triangle
        # assert attention_mask is None, f"{self.__class__.__name__} does not support attention masks right now"
        logger.warning(f"forward: {input_ids.shape=} {session=} {kwargs=}")

        for k, v in kwargs.items():
            if not (v is None or v is False):
                logger.warning(f"Extra keyword arguments are not yet supported (got {k} = {v})")

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
            inputs_embeds = self.embed_tokens(input_ids)

        if (
            self.config.tuning_mode
            and "ptune" in self.config.tuning_mode
            and (session is None or session.position == 0)
        ):
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            prompts = intermediate_prompts = None

        hidden_states = inputs_embeds
        output_shape = input_shape + (hidden_states.size(-1),)

        if session is not None:
            hidden_states = session.step(
                hidden_states,
                prompts=intermediate_prompts,
                hypo_ids=past_key_values.hypo_ids if past_key_values is not None else None,
            )
        else:
            hidden_states = self.layers(hidden_states, prompts=intermediate_prompts)

        # Remove prefix
        if self.config.tuning_mode and "ptune" in self.config.tuning_mode:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        # Add last hidden state
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=RemotePastKeyValues(),
            hidden_states=None,
            attentions=None,
        )

    @property
    def word_embeddings(self) -> nn.Embedding:  # For compatibility with RemoteGenerationMixin
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:  # For compatibility with RemoteGenerationMixin
        return self.layers

    @property
    def ln_f(self) -> nn.Module:  # For compatibility with RemoteGenerationMixin
        return self.norm


class DistributedLlamaForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, LlamaForCausalLM):
    _keys_to_ignore_on_load_missing = DistributedLlamaModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedLlamaModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedLlamaConfig

    def __init__(self, config: DistributedLlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = DistributedLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        session: Optional[InferenceSession] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            session=session,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedLlamaModel:  # For compatibility with RemoteGenerationMixin
        return self.model


class DistributedLlamaForSequenceClassification(FromPretrainedMixin, LlamaForSequenceClassification):
    _keys_to_ignore_on_load_missing = DistributedLlamaModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedLlamaModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedLlamaConfig

    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        self.model = DistributedLlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def transformer(self) -> DistributedLlamaModel:  # For compatibility with RemoteGenerationMixin
        return self.model
