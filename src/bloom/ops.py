"""
Utility operations used in the the BLOOM model
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""
import math

import torch
import torch.autograd
from torch import nn


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.

    Args:
        tensor: ([`torch.tensor`], *required*):
            input tensor to split
        num_partitions ([`int`], *required*):
            number of partitions to split the tensor
        contiguous_split_chunks ([`bool`], *optional*, default=`False`)::
            If True, make each chunk contiguous in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    numerator, denominator = tensor.size()[last_dim], num_partitions
    if not (numerator % denominator == 0):
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    last_dim_size = numerator // denominator
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def attention_mask_func(attention_scores, attention_mask, causal_mask):
    if attention_mask.dtype == torch.bool:
        attention_mask_bool = ~attention_mask
    else:
        attention_mask_bool = (1 - attention_mask).bool()

    query_length, key_length, n_heads = attention_scores.size(2), attention_scores.size(3), attention_scores.size(1)
    padded_causal_mask = (
        attention_mask_bool[:, None, key_length - query_length : key_length, None]
        + ~causal_mask[:, :, key_length - query_length : key_length, :key_length]
    ).bool()
    padded_causal_mask = padded_causal_mask + attention_mask_bool[:, None, None, :key_length].bool()
    # Make use of floats
    return (
        attention_scores.masked_fill_(padded_causal_mask.expand(-1, n_heads, -1, -1), -10000.0),
        padded_causal_mask,
    )


def build_alibi_tensor(max_seq_len, n_head, dtype=torch.bfloat16):
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742

    Args:
    Returns tensor shaped (n_head, 1, max_seq_len)
        max_seq_len: (`int`, *required*):
            max sequence length
        n_head: (`int`, *required*):
            number of heads
        dtype: (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.Tensor(get_slopes(n_head)).unsqueeze(1).unsqueeze(1)
    arange_tensor = torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0)
    alibi = slopes * arange_tensor.expand(n_head, -1, -1)

    alibi = alibi.to(dtype)

    return alibi


def pre_process_alibi_for_pad(alibi, attention_mask, num_heads):
    """
    Args:
    Pre-process the alibi tensor for padding.
        alibi: ([`torch.tensor`], *required*):
            alibi tensor to pre-process
        attention_mask: ([`torch.tensor`], *required*):
            attention mask to pre-process"""

    # Sanity check if we are not inferring less tokens than the total sequence length
    # This usually happens when the inference is done with past_key_values
    # In this case we re-create the alibi tensor with the correct sequence length
    if attention_mask.shape[-1] != alibi.shape[-1]:
        alibi = build_alibi_tensor(attention_mask.shape[-1], num_heads, alibi.dtype).repeat(
            attention_mask.shape[0], 1, 1
        )
    # Get the indexes of the padding tokens
    index_x0, index_y0 = torch.where(attention_mask == 0.0)
    index_x1, index_y1 = torch.where(attention_mask == 1.0)

    # Clone the embeddings  - we can detach because the embeddings are not learned
    # Get a refence tensor
    slice_reference_alibi = build_alibi_tensor(alibi.shape[-1], num_heads, alibi.dtype)

    # Loop over the batch where the padding is and replace the alibi tensor by the reference tensor
    # Only where you do not have padding. Replace padding tokens by zeros
    # This operation can be seen as a shifting operation.
    for i, index in enumerate(torch.unique(index_x0)):
        slice_to_modify = torch.zeros_like(slice_reference_alibi)
        index_shift = index_y1[index_x1 == index]
        shift_value = len(index_shift)
        slice_to_modify[:, :, index_shift] = slice_reference_alibi[:, :, :shift_value]
        alibi[index * num_heads : (index + 1) * num_heads] = slice_to_modify
    return alibi


def dropout_add(x, residual, prob, training):
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *rquired*):
            esidual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def bloom_gelu_forward(x):
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def bloom_gelu_back(g, x):
    """
    gradient of tanh approximation of gelu gradient of actual gelu is: 0.5 * (1. + torch.erf(x * 0.70710678)) +
    0.3989423 * x * torch.exp(-0.5 * x * x)

    Args:
        g (`torch.tensor`, *required*):
            gradient output tensor
        x (`torch.tensor`, *required*):
            input tensor
    """
    x = x[0]  # x is a tuple of 1 element, needs to unpack it first
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = bloom_gelu_back(grad_output, input)
        return tmp


class BloomGelu(nn.Module):
    """
    BloomBiasGelu wrapper function that make use of the simple function on inference mode to make the model
    torchscriptable and use the autograd function in training mode to get the accurate results of the gradients Partly
    copied from Megatron-DeepSpeed code and adapted for our needs

    See here why autograd functions are not torchscriptable: https://github.com/pytorch/pytorch/issues/22329

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            return GeLUFunction.apply(x)
        else:
            return bloom_gelu_forward(x)


class BloomScaledSoftmax(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Args:
        input_in_fp16 (`bool`, *required*):
            flag to indicate if input in fp16 data format.
        input_in_bf16 (`bool`, *required*):
            flag to indicate if input in bf16 data format.
        scaled_masked_softmax_fusion (`bool`, *required*):
            flag to indicate user want to use softmax fusion
        mask_func (`function`, *required*):
            mask function to be applied.
        softmax_in_fp32 (`bool`, *required*):
            if true, softmax in performed at fp32 precision.
        scale (`float`, *required*):
            scaling factor used in input tensor scaling.
    """

    def __init__(self, scaled_masked_softmax_fusion, mask_func, softmax_in_fp32, scale):
        super().__init__()
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        if not (self.scale is None or softmax_in_fp32):
            raise ValueError("softmax should be in fp32 when scaled")

    def forward(self, input, mask, max_positions):
        input_dtype = input.dtype
        input_in_16bit = input_dtype in [torch.float16, torch.bfloat16]
        softmax_dtype = torch.float32 if self.softmax_in_fp32 else input_dtype

        if self.scale is not None:
            input = input * self.scale

        if mask is not None:
            mask = mask.to(input.device)
            causal_mask = (
                torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool))
                .view(1, 1, max_positions, max_positions)
                .to(input.device)
            )
            mask_output, padded_causal_mask = self.mask_func(input, mask, causal_mask)
            probs = nn.functional.softmax(mask_output, dim=-1, dtype=softmax_dtype) * (~padded_causal_mask)
        else:
            probs = nn.functional.softmax(input, dim=-1, dtype=softmax_dtype)

        if input_in_16bit and self.softmax_in_fp32:
            probs = probs.to(dtype=input_dtype)

        return probs
