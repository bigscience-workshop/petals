import contextlib
from typing import Optional

from hivemind.utils.logging import get_logger

from petals.client.inference_session import InferenceSession

logger = get_logger(__name__)


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

    def generate(self, *args, session: Optional[InferenceSession] = None, **kwargs):
        if session is None:
            context_manager = self.inference_session(max_length=2048)  # FIXME: Provide actual length
        else:
            context_manager = contextlib.nullcontext(session)  # Doesn't actually enter session or exit from it
        with context_manager as session:
            return super().generate(*args, session=session, **kwargs)
