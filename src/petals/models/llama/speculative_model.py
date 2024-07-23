from typing import Optional, Union

import torch

from transformers.generation import LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation.utils import GenerateNonBeamOutput, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaForCausalLM

from petals.models.llama.config import DistributedLlamaConfig
from petals.models.llama.model import DistributedLlamaForCausalLM


class DistributedLlamaForSpeculativeGeneration(DistributedLlamaForCausalLM, GenerationMixin):
    def __init__(self, config: DistributedLlamaConfig, small_model: LlamaForCausalLM):
        DistributedLlamaForCausalLM.__init__(self, config)
        self.small_model = small_model

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        logits_warper: Optional[LogitsProcessorList],
        speculative_batch_size: int = 10,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        print(model_kwargs)

        pad_token_id = generation_config.pad_token_id
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        
        assert not generation_config.do_sample, "sample is not working for speculative generation now"

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        firsts = True

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            speculative_batch_size = min(speculative_batch_size, self.active_session._max_length - input_ids.shape[1])
            with torch.no_grad():
                speculative_outputs = self.small_model.generate(
                    input_ids,
                    max_new_tokens=speculative_batch_size, 
                    do_sample=False,
                    use_cache=False
                )
                speculative_tokens = speculative_outputs[:, -speculative_batch_size:]

            full_sequence = torch.cat([input_ids, speculative_tokens], dim=-1)
            assert input_ids.shape[1] + speculative_batch_size == full_sequence.shape[1]

            with torch.no_grad():
                real_input = full_sequence
                if not firsts:
                    self.active_session.position = input_ids.shape[1] - 1
                    real_input = real_input[:, -speculative_batch_size - 1:]
                else:
                    firsts = False
                real_input = real_input[:, :-1]

                precise_model_outputs = self(real_input, return_dict=True)
                full_token_logits = precise_model_outputs.logits[:, -speculative_batch_size:, :].clone()

            all_valid_tokens = []

            first_token = None
            for i in range(speculative_batch_size):
                token_logits = full_token_logits[:, i, :]
                valid_token = torch.argmax(token_logits, dim=-1)

                if first_token is None:
                    first_token = valid_token

                if valid_token.item() == speculative_tokens[:, i].item():
                    all_valid_tokens.append(valid_token.unsqueeze(-1))
                else:
                    break

            if not all_valid_tokens and first_token is not None:
                all_valid_tokens.append(first_token.unsqueeze(-1))
            all_valid_tokens = torch.cat(all_valid_tokens, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                all_valid_tokens = all_valid_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, all_valid_tokens], dim=-1)

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0

            del precise_model_outputs

        return input_ids
