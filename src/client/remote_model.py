# this code is in active development, interfaces may change
import os
from typing import Optional, Union, Tuple

import hivemind
from hivemind import DHT, get_logger, use_hivemind_log_handler

from src.bloom import BloomForYou, DistributedBloomConfig
from src.bloom.from_pretrained import CLIENT_BRANCH, _load_state_dict
from src.client.remote_sequential import RemoteSequential
from src.data_structures import UID_DELIMITER

import torch
from hivemind import use_hivemind_log_handler
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


class DistributedBloomForYou(BloomForYou):
    """BloomModel, but all transformer layers are hosted by the swarm"""

    def __init__(self, config: DistributedBloomConfig, dht: DHT, prefix: str):
        n_layer, config.n_layer = config.n_layer, 0  # temporarily set n_layer to 0 to prevent layer initialization
        super().__init__(config)
        assert len(self.transformer.h) == 0
        config.n_layer = n_layer
        self.transformer.h = RemoteSequential(config, dht, prefix)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        if "initial_peers" not in kwargs:
            raise ValueError("Please specify initial_peers=...")

        dht = hivemind.DHT(
            initial_peers=kwargs.pop("initial_peers"), client_mode=kwargs.pop("client_mode", True), start=True
        )

        if "prefix" not in kwargs:
            logger.debug(f"No DHT prefix specified; using automatic prefix {pretrained_model_name_or_path}")
            assert (
                UID_DELIMITER not in pretrained_model_name_or_path
            ), f"Cannot infer prefix automatically from {pretrained_model_name_or_path}; please specify prefix=..."
        prefix = kwargs.pop("prefix", pretrained_model_name_or_path)

        config = DistributedBloomConfig.from_pretrained(pretrained_model_name_or_path, revision=CLIENT_BRANCH, **kwargs)
        model = cls(config, dht, prefix)
        model.transformer.load_state_dict(
            _load_state_dict(pretrained_model_name_or_path, use_auth_token=kwargs.get("use_auth_token")), strict=True
        )
        return model


class DistributedBloomForCausalLM(DistributedBloomForYou):
    """DistributedBloomForCausalLM, but all transformer layers are hosted by the swarm"""

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def forward(self, input_ids, labels=None, return_dict=None, **kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer.forward(input_ids=input_ids, return_dict=return_dict, **kwargs)

        # Switch dtype in case word_embeddings are fp16
        word_embeddings = self.transformer.word_embeddings.weight.t()
        hidden_states = transformer_outputs[0].to(word_embeddings.dtype)
        lm_logits = (hidden_states @ word_embeddings).float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
