import os

import bitsandbytes as bnb
import torch

PETALS_8BIT_BACKWARD = bool(int(os.environ.get("PETALS_8BIT_BACKWARD", 0)))


def replace_8bit_linear(model, threshold=6.0):
    """
    A helper function to convert all `torch.nn.Linear` modules to `bnb.nn.Linear8bit` modules from the `bitsandbytes`
    library. This will enable running your models using mixed int8 precision as described by the paper `GPT3.int8():
    8-bit Matrix Multiplication for Transformers at Scale`. Make sure `bitsandbytes` compiled with the correct CUDA
    version of your hardware is installed before running this function. `pip install -i https://test.pypi.org/simple/
    bitsandbytes-cudaXXX` with `XXX` is your CUDA version (e.g., 11.6 = 116)
    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` and 'score' that should
    be kept as a `torch.nn.Linear` module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        threshold (`float`, *optional*):
            `int8_threshold` for outlier detection as described in the formentioned paper. This parameters is set to
            `6.0` as described by the paper.
    """
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module, threshold)

        if isinstance(module, torch.nn.Linear) and n not in ["lm_head", "score"]:
            model._modules[n] = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                module.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
                memory_efficient_backward=PETALS_8BIT_BACKWARD,
            )
            model._modules[n].weight = bnb.nn.Int8Params(
                module.weight.data, requires_grad=False, has_fp16_weights=False
            ).to(module.weight.dtype)
    return model
