import dataclasses
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from hivemind import get_logger
from transformers import PretrainedConfig

from petals.utils.misc import DUMMY

logger = get_logger(__name__)


@dataclasses.dataclass
class PTuneConfig:
    pre_seq_len: int = 0  # a number of tokens for prompt tuning.
    tuning_mode: Optional[str] = None  # fine-tuning regime, one of [None, "ptune", "deep_ptune"]


class PTuneMixin:
    _keys_to_ignore_on_load_missing = [r"(intermediate_)?prompt_embeddings\.weight$"]

    def init_prompts(self, config: PretrainedConfig) -> None:
        if config.tuning_mode and "ptune" in config.tuning_mode:
            assert config.pre_seq_len > 0, "The number of prefix tokens must be > 0"
            self.pre_seq_len = config.pre_seq_len
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()

            with force_non_empty_weights():
                # Prompt embeddings and their optimizer stats are kept in float32 to increase ptune quality
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

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        prefix_tokens = prefix_tokens.to(self.word_embeddings.weight.device)
        prompts = self.prompt_embeddings(prefix_tokens)

        if self.config.tuning_mode == "deep_ptune":
            intermediate_prompts = self.intermediate_prompt_embeddings(prefix_tokens)
            intermediate_prompts = intermediate_prompts.view(
                batch_size,
                self.pre_seq_len,
                self.config.num_hidden_layers,
                self.config.hidden_size
                # TODO: should be num_hidden_layers - 1
            )
            intermediate_prompts = intermediate_prompts.permute([2, 0, 1, 3])
        else:
            intermediate_prompts = DUMMY

        dtype = self.word_embeddings.weight.dtype
        return prompts.to(dtype), intermediate_prompts.to(dtype)


_original_register_parameter = nn.Module.register_parameter


@contextmanager
def force_non_empty_weights():
    """
    This context manager allows to bypass the accelerate.init_empty_weights() context manager
    (that forces all nn.Parameters to be PyTorch's meta tensors) used when low_cpu_mem_usage=True.
    The transformers library should replace all meta tensors by empty tensors by itself
    but this feature does not work due to a bug ([1] fails if `add_prefix_to_model == True`).

    [1] https://github.com/huggingface/transformers/blob/ab9fe45236cd99b8797df78219438f8f6662bb42/src/transformers/modeling_utils.py#L2515
    """

    possibly_patched_register_parameter = nn.Module.register_parameter
    nn.Module.register_parameter = _original_register_parameter
    try:
        yield
    finally:
        nn.Module.register_parameter = possibly_patched_register_parameter
