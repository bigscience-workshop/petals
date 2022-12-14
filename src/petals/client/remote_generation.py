import contextlib
from typing import List, Optional

import torch
from hivemind.utils.logging import get_logger

from petals.client.inference_session import InferenceSession
from petals.utils.generation_algorithms import (
    BeamSearchAlgorithm,
    DecodingAlgorithm,
    GreedyAlgorithm,
    NucleusAlgorithm,
    SamplingAlgorithm,
    TopKAlgorithm,
)
from petals.utils.generation_constraints import ABCBloomConstraint, EosConstraint

logger = get_logger(__file__)


class RemoteGenerationMixin:
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in [`BloomForCausalLM`].
    The class exposes can be used for:
        - *greedy decoding*.
        - *multinomial, top-k and top-p sampling*.
        - *beam-search decoding*

    This class is similar to transformer's [`generation_utils.GenerationMixin`], it can be used instead of it.
    However, it has some differences for remote usage.
    """

    def inference_session(self, **kwargs) -> InferenceSession:
        """
        Returns an inference session for the model's RemoteSequential module.

        :param max_length: Maximal expected length of inference results. Servers use this parameter
                           to calculate the size of attention caches allocated to this client.
        """

        return self.transformer.h.inference_session(**kwargs)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        do_sample: Optional[bool] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_beams: Optional[int] = 1,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        decoding_algorithm: Optional[DecodingAlgorithm] = None,
        provided_constraints: List[ABCBloomConstraint] = [],
        num_return_sequences: Optional[int] = None,
        *,
        session: Optional[InferenceSession] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        """
        Generates sequences of token ids for models with a language modeling head.

        :param inputs: The input tokens to the model.
        :param do_sample: Whether to sample from the model predictions or take the argmax.
        :param temperature: The temperature to use for sampling.
        :param top_k: The number of results to return.
        :param top_p: The cumulative probability of results to return.
        :param num_beams: The number of beams to use for beam search.
        :param bos_token_id: The id of the beginning of sentence token.
        :param eos_token_id: The id of the end of sentence token.
        :param pad_token_id: The id of the padding token.
        :param max_length: The maximum number of tokens in the output (including input tokens).
        :param max_new_tokens: The maximum number of tokens to generate.
        :param decoding_algorithm: The decoding algorithm to use.
        :param provided_constraints: A list of constraints to use.
        :param model_kwargs: Additional arguments to pass to the model.
        :param num_return_sequences: How many hypothesis from the beam will be in output.
        """

        assert (
            model_kwargs.get("logits_processor", None) is None
        ), "For RemoteGenerationMixin models use BloomConstraints instead of logits_processor"
        assert (
            model_kwargs.get("logits_wrapper", None) is None
        ), "For RemoveGenerationMixin models use DecodingAlgorithm instead of logits_wrapper"
        assert (
            model_kwargs.get("stopping_criteria", None) is None
        ), "For RemoteGenerationMixin models use BloomConstraints instead of stopping_criteria"
        prefix_length = 0 if inputs is None else inputs.size(1)
        prefix_length += self.config.pre_seq_len

        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        assert (max_length is None) != (max_new_tokens is None), "please set max_length or max_new_tokens (not both)"
        if max_length is not None and max_new_tokens is None:
            max_new_tokens = max_length - prefix_length
            assert max_new_tokens > 0, f"Provided max_length is less than prefix size: {max_length} < {inputs.size(1)}"
        elif max_length is None and max_new_tokens is not None:
            max_length = prefix_length + max_new_tokens

        if num_beams > 1 and session is not None:
            raise NotImplementedError(
                "Reusing inference session in .generate() along with beam search is not supported yet"
            )

        if inputs is not None:
            assert isinstance(inputs, torch.Tensor) and inputs.ndim == 2, "inputs must be a 2d tensor [batch, length]"
            if session is not None and session.last_token_id is not None:
                inputs = torch.cat([session.last_token_id, inputs], dim=1)
        else:
            if session is not None and session.last_token_id is not None:
                inputs = session.last_token_id
            else:
                assert bos_token_id is not None, "You have to provide a bos_token_id if you do not provide inputs"
                inputs = torch.tensor([[bos_token_id]] * num_beams, dtype=torch.long, device=self.device)
        batch_size = inputs.size(0)

        if decoding_algorithm is None:
            if do_sample:
                decoding_algorithm = self._choose_sample_algorithm(temperature, top_k, top_p)
            elif num_beams is not None and num_beams > 1:
                decoding_algorithm = BeamSearchAlgorithm(num_beams, batch_size=batch_size)
            else:
                if top_k is not None or top_p is not None:
                    logger.warning("You passed top_k or top_p but did pass do_sample=True. Running greedy sampling")
                decoding_algorithm = GreedyAlgorithm()

        if num_beams > 1:
            inputs = torch.cat([inputs] * num_beams, dim=0)
            if batch_size > 1:
                # TODO: resolve padding problem
                logger.warning(
                    f"You set batch_size {batch_size} within beam search generation. "
                    f"Be careful, results on sequences with different length may be padded wrong way"
                )

        if num_return_sequences is None:
            num_return_sequences = 1

        assert num_return_sequences <= num_beams, (
            f"You want more sequences than the beam has."
            " Check num_return_sequences: {num_return_sequences} and num_beams: {num_beams}."
        )

        constraints = self._get_constraints(
            inputs=inputs,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            provided_constraints=provided_constraints,
        )

        if session is None:
            context_manager = self.inference_session(max_length=max_length)
        else:
            context_manager = contextlib.nullcontext(session)  # Doesn't actually enter session or exit from it
        with context_manager as session:
            outputs = []
            # Find samples with padded inputs.
            # They will be changed before all of the samples have right length.
            if torch.any(inputs == pad_token_id):  # TODO: move to prepare_inputs
                outputs += [inputs[:, : inputs.size(1) - (inputs == pad_token_id).sum(-1).max()]]
            else:
                outputs += [inputs]
            last_token_id = None
            seq_idx = outputs[0].size(1)
            hypo_ids = torch.arange(outputs[0].size(0))
            while True:
                embs = self.transformer.word_embeddings(outputs[-1])
                intermediate_prompts = None
                if self.config.pre_seq_len > 0 and len(outputs) == 1:
                    prompts, intermediate_prompts = self.transformer.get_prompt(embs.size(0))
                    embs = torch.cat([prompts, embs], dim=1)
                embs = self.transformer.word_embeddings_layernorm(embs)
                hidden_state = session.step(embs, prompts=intermediate_prompts, hypo_ids=hypo_ids)[:, -1]
                hidden_state = self.transformer.ln_f(hidden_state)
                lm_logits = self.lm_head(hidden_state)

                for constraint in constraints:
                    lm_logits = constraint(last_token_id, lm_logits, hypo_ids)
                last_token_id, hypo_ids = decoding_algorithm(lm_logits)

                # If some samples were padded, change only these samples
                if seq_idx < inputs.size(1):
                    pad_token_mask = inputs[:, seq_idx : seq_idx + 1] == pad_token_id
                    last_token_id = (~pad_token_mask) * inputs[
                        :, seq_idx : seq_idx + 1
                    ] + pad_token_mask * last_token_id

                # TODO: refactor outputs
                if num_beams > 1:
                    for i in range(len(outputs), 1, -1):
                        outputs[i - 1] = outputs[i - 1][hypo_ids]

                outputs.append(last_token_id)
                session.last_token_id = last_token_id
                seq_idx += 1
                if torch.all(last_token_id == eos_token_id) or len(outputs) > max_new_tokens:
                    break

        outputs = torch.cat(outputs, dim=-1)

        if num_beams > 1:
            pre_return_idx = [
                torch.arange(idx, num_return_sequences * batch_size, batch_size) for idx in range(batch_size)
            ]
            return_idx = torch.cat(pre_return_idx, dim=0)
            outputs = outputs[return_idx]

        return outputs

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        provided_constraints: List[ABCBloomConstraint] = [],
        **model_kwargs,
    ) -> torch.LongTensor:
        """
        Generates sequences of token ids for models with a language modeling head. Uses greedy search.

        :param input_ids: The input tokens to the model.
        :param max_length: The maximum length of the sequence to generate.
        :param pad_token_id: The id of the padding token.
        :param eos_token_id: The id of the end of sentence token.
        :param provided_constraints: A list of constraints to use.
        """
        return self.generate(
            inputs=input_ids,
            max_new_tokens=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoding_algorithm=GreedyAlgorithm(),
            provided_constraints=provided_constraints,
            **model_kwargs,
        )

    def sample(
        self,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        provided_constraints: List[ABCBloomConstraint] = [],
        **model_kwargs,
    ) -> torch.LongTensor:
        """
        Generates sequences of token ids for models with a language modeling head. Uses multinomial sampling.
        If top_k is provided, uses top_k sampling. If top_p is provided, uses nucleus sampling.

        :param: input_ids: The input tokens to the model.
        :param: temperature: The temperature to use for sampling.
        :param: top_k: The number of samples to use for top_k sampling.
        :param: top_p: The probability of using top_p sampling.
        :param: max_length: The maximum length of the sequence to generate.
        :param: pad_token_id: The id of the padding token.
        :param: eos_token_id: The id of the end of sentence token.
        :param: provided_constraints: A list of constraints to use.
        :param: model_kwargs: Additional kwargs to pass to the model.
        """

        return self.generate(
            inputs=input_ids,
            max_new_tokens=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoding_algorithm=self._choose_sample_algorithm(temperature, top_k, top_p),
            provided_constraints=provided_constraints,
            **model_kwargs,
        )

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        num_beams: int = 1,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        provided_constraints: List[ABCBloomConstraint] = [],
        **model_kwargs,
    ) -> torch.LongTensor:
        """
        Generates sequences of token ids for models with a language modeling head. Uses beam search.

        :param input_ids: The input tokens to the model.
        :param num_beams: The number of beams to use.
        :param max_length: The maximum length of the sequence to generate.
        :param pad_token_id: The id of the padding token.
        :param eos_token_id: The id of the end of sentence token.
        :param provided_constraints: A list of constraints to use.
        :param: model_kwargs: Additional kwargs to pass to the model.
        """
        decoding_algorithm = BeamSearchAlgorithm(
            num_beams=num_beams,
            batch_size=input_ids.size(0),
        )
        return self.generate(
            inputs=input_ids,
            num_beams=num_beams,
            max_new_tokens=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoding_algorithm=decoding_algorithm,
            provided_constraints=provided_constraints,
            **model_kwargs,
        )

    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        provided_constraints: List[ABCBloomConstraint] = [],
        **model_kwargs,
    ) -> torch.LongTensor:
        raise NotImplementedError

    def group_beam_search(
        self,
        input_ids: torch.LongTensor,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        provided_constraints: List[ABCBloomConstraint] = [],
        **model_kwargs,
    ) -> torch.LongTensor:
        raise NotImplementedError

    def _choose_sample_algorithm(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> DecodingAlgorithm:
        if (top_k is not None) and (top_p is not None):
            raise ValueError("You have to provide only top_k or top_p for sampling")
        if top_k is not None:
            return TopKAlgorithm(top_k, temperature)
        elif top_p is not None:
            return NucleusAlgorithm(top_p, temperature)
        else:
            return SamplingAlgorithm(temperature)

    def _get_constraints(
        self,
        inputs: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        provided_constraints: List[ABCBloomConstraint] = [],
    ) -> List[ABCBloomConstraint]:
        constraints = []
        constraints.extend(provided_constraints)
        constraints.append(EosConstraint(inputs, eos_token_id, pad_token_id))
        return constraints
