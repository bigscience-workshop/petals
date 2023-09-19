import dataclasses
import platform
from typing import Optional, Union

import psutil
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from hivemind import get_logger
from torch import nn
from transformers import PretrainedConfig

logger = get_logger(__name__)


@dataclasses.dataclass
class LMHeadConfig:
    # This settings matter for running the client with dtype bfloat16 on CPU.
    # If the CPU doesn't support AVX512, chunked_forward() significantly speeds up computations.
    use_chunked_forward: Union[str, bool] = "auto"
    chunked_forward_step: int = 16384


class LMHead(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()

        if not config.tie_word_embeddings:
            self.weight = nn.Parameter(torch.zeros(config.vocab_size, config.hidden_size))
            self.weight.requires_grad = False
        else:
            self.weight = None  # Will be set to get_input_embeddings().weight during loading the model
        self.bias = None
        self.in_features = config.hidden_size  # Similar to nn.Linear attributes
        self.out_features = config.vocab_size

        self.use_chunked_forward = config.use_chunked_forward
        if self.use_chunked_forward == "auto":
            if platform.machine() == "x86_64":
                # Import of cpufeature may crash on non-x86_64 machines
                from cpufeature import CPUFeature

                # If the CPU supports AVX512, plain bfloat16 is ~10x faster than chunked_forward().
                # Otherwise, it's ~8x slower.
                self.use_chunked_forward = not (CPUFeature["AVX512f"] and CPUFeature["OS_AVX512"])
            else:
                self.use_chunked_forward = True
        self.chunked_forward_step = config.chunked_forward_step
        self._bf16_warning_shown = False

    def forward(self, hidden_states):
        if (
            self.weight.dtype in [torch.float16, torch.bfloat16]
            and self.weight.device.type == "cpu"
            and self.use_chunked_forward
        ):
            lm_logits = self.chunked_forward(hidden_states)
        else:
            # Switch dtype in case word_embeddings are fp16/bf16
            hidden_states = hidden_states.to(self.weight.dtype)
            lm_logits = F.linear(hidden_states, self.weight)
        return lm_logits

    def chunked_forward(self, hidden_states):
        """Splits word embeddings on chunks and iteratively casts them into fp32 to perform matmul more efficiently on CPU.
        chunked_forward_step: provides trade-off between efficiency and extra memory consumption.
        """
        assert self.chunked_forward_step > 0, "Chunk size for chunked forward must be positive"

        if not self._bf16_warning_shown:
            if self.weight.numel() * 4 < 0.9 * psutil.virtual_memory().total:
                logger.warning(
                    "Running the model in bfloat16 on CPU will be slow since your CPU does not support AVX512. "
                    "To speed it up, load the model in float32 using .from_pretrained(..., torch_dtype=torch.float32)"
                )
            self._bf16_warning_shown = True

        hidden_states = hidden_states.float()
        output = torch.empty(*hidden_states.shape[:-1], self.out_features)

        for i in range(0, self.out_features, self.chunked_forward_step):
            chunk = self.weight[i : i + self.chunked_forward_step].float()
            output[..., i : i + self.chunked_forward_step] = F.linear(hidden_states, chunk)
        return output
