"""
Utility operations used in the the BLOOM model
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""
import math

import torch
import torch.autograd
import torch.nn.functional as F
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


def build_alibi_tensor(
    max_seq_len: int, n_head: int, dtype: torch.dtype = torch.bfloat16, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
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
        device: (`torch.device`, *optional*, default=`torch.device('cpu')`):
            device of the output alibi tensor
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
    base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=device, dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2, device=device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != n_head:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, n_head - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    lengths = torch.arange(max_seq_len, device=device, dtype=torch.int32)
    return (slopes.view(-1, 1, 1) * lengths.view(1, 1, -1)).to(dtype)


def pre_process_alibi_for_pad(alibi: torch.Tensor, attention_mask: torch.Tensor):
    """
    Args:
    Pre-process the alibi tensor for padding.
        alibi: ([`torch.tensor`], *required*):
            alibi tensor to pre-process
        attention_mask: ([`torch.tensor`], *required*):
            attention mask to pre-process
    """
    assert attention_mask.ndim == 2, "mask should be [batch_size, seq_length]"
    unpadded_indices = torch.relu(attention_mask.cumsum(dim=1) - 1)
    # ^-- [batch, max_len], values correspond to element indices after removing padding
    # We shift the alibi tensor + replace all the values where attention_mask==0.0 by 0
    alibi = alibi.take_along_dim(unpadded_indices.unsqueeze(0), -1) * attention_mask.unsqueeze(0)
    return alibi.reshape(alibi.shape[0] * alibi.shape[1], 1, -1)


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

        if mask is None:
            mask = torch.ones(input.shape[0], max_positions, dtype=torch.bool, device=input.device)

        mask = mask.to(input.device)
        causal_mask = (
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool))
            .view(1, 1, max_positions, max_positions)
            .to(input.device)
        )
        mask_output, padded_causal_mask = self.mask_func(input, mask, causal_mask)
        probs = F.softmax(mask_output, dim=-1, dtype=softmax_dtype) * (~padded_causal_mask)

        if input_in_16bit and self.softmax_in_fp32:
            probs = probs.to(dtype=input_dtype)

        return probs
