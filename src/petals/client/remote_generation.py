import contextlib
import dataclasses
from typing import ContextManager, List, Optional

import torch
import transformers
from hivemind.utils.logging import get_logger
from transformers.generation.utils import ModelOutput

from petals.client.inference_session import InferenceSession
from petals.client.remote_sequential import RemoteSequential
from petals.utils.misc import DUMMY, docstring_from

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class RemotePastKeyValues:
    """A mock class representing the fact that `past_key_values` do exist but are stored on remote servers."""

    hypo_ids: Optional[torch.LongTensor] = None

    def __getitem__(self, _index: int) -> List[torch.Tensor]:
        return [DUMMY]  # For compatibility with BloomForCausalLM.prepare_inputs_for_generation()


class RemoteGenerationMixin:
    """
    This class is an upgrade to `transformers.GenerationMixin` that:

    - Designed to be compatible with most `transformers.GenerationMixin` strategies and options
    - Supports generation inside a remote InferenceSession, so that remote servers store your attention caches and
      you don't have to rerun the prefix through all the servers to generate each new token
    - Supports multiple `.generate()` calls inside one InferenceSession, so you can easily run interactive generation
      by showing tokens on the fly (multiple calls like `.generate(None, max_new_tokens=1, ...)`) or
      accept prompts from a user in a chat bot (multiple calls like `.generate(new_prompts, ...)`).
    - If there is no active session, `.generate()` will create a new InferenceSession with proper `max_length`.
      Otherwise, `.generate()` will use the active session. You can use the `session=...` argument to override that.
    """

    @docstring_from(RemoteSequential.active_session)
    @property
    def active_session(self) -> Optional[InferenceSession]:
        return self.transformer.h.active_session

    @docstring_from(RemoteSequential.use_session)
    def use_session(self, session: Optional[InferenceSession]) -> ContextManager[InferenceSession]:
        return self.transformer.h.use_session(session)

    @docstring_from(RemoteSequential.inference_session)
    def inference_session(self, **kwargs) -> ContextManager[InferenceSession]:
        return self.transformer.h.inference_session(**kwargs)

    @docstring_from(transformers.GenerationMixin.generate.__doc__)
    def generate(
        self, inputs: Optional[torch.Tensor] = None, *args, session: Optional[InferenceSession] = None, **kwargs
    ):
        if session is not None:
            # If a session specified explicitly, use it
            context_manager = self.use_session(session)
        elif self.active_session is not None:
            # If there's an active session, don't do anything
            context_manager = contextlib.nullcontext(self.active_session)
        else:
            # If there's no active session, create a new one

            max_length = kwargs.get("max_length")
            max_new_tokens = kwargs.get("max_new_tokens")
            assert (max_length is None) != (
                max_new_tokens is None
            ), "You should set `max_length` or `max_new_tokens` (but not both) to reserve server-side attention caches"

            if max_length is not None:
                session_max_length = max_length
            else:
                session_max_length = (inputs.shape[1] if inputs is not None else 0) + max_new_tokens
            context_manager = self.inference_session(max_length=session_max_length)

        with context_manager as session:
            # Prepend the last tokens from the previous .generate() call
            if session.last_token_id is not None:
                assert session.last_token_id.shape[1] == 1, f"{session.last_token_id.shape=} is invalid"
                if inputs is not None:
                    inputs = torch.cat([session.last_token_id, inputs], dim=1)
                else:
                    inputs = session.last_token_id

            result = super().generate(inputs, *args, **kwargs)

            sequences = result.sequences if isinstance(result, ModelOutput) else result
            # Crop the last tokens from the previous call
            if session.last_token_id is not None:
                sequences = sequences[:, 1:]
                if isinstance(result, ModelOutput):
                    result.sequences = sequences
                else:
                    result = sequences
            # Save the last tokens from this call
            session.last_token_id = sequences[:, -1:]

        return result

    @staticmethod
    def _reorder_cache(past_key_values: RemotePastKeyValues, beam_idx: torch.LongTensor) -> RemotePastKeyValues:
        return dataclasses.replace(past_key_values, hypo_ids=beam_idx)
