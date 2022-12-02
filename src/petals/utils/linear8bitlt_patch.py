"""
A patch to bitsandbytes 0.34.0 that introduces an option to run backward pass in default (fast) matrix layout
Authors: modification by @borzunov, original code by @timdettmers. Please disregard commit authors in this file.

Core idea: layouts apply the same permutation to every tile in the matrix. We can treat this as (batched) gather ops.
  Reshape input tensor so that ij-th gather operation op will apply to ij-th elements in each tile.
Prototype: https://colab.research.google.com/drive/1EJ0MKifajXSSVq7O2_QGwtb0l6gRAGrh?usp=sharing
Based on: https://github.com/TimDettmers/bitsandbytes/blob/main/csrc/kernels.cu#L2130-L2136
Exact match tests: see $REPO/tests/test_linear8bitlt.py
"""
import dataclasses
from typing import Optional, Tuple

import bitsandbytes.functional as F
import torch
from bitsandbytes.autograd._functions import MatMul8bitLt, MatmulLtState
from bitsandbytes.nn import Linear8bitLt


def get_inverse_transform_indices(transform_tile: callable, tile_size: Tuple[int, int]):
    """
    Compute a permutation of indices that invert the specified (tiled) matrix transformation

    :param transform_tile: a function that applies forward transform to a tensor of shape [dim1, dim2]
    :param tile_size: higher-level tile dimensions, i.e. (8, 32) for Turing and (32, 32) for Ampere
    :note: we assume that tile_transform applies to a cpu-based int8 tensor of shape tile_size
    :example: transform_tile function for the turing layout (bitsandbytes.functional as F)
    :returns: indices
    """
    d1, d2 = tile_size
    assert 0 < d1 * d2 < 2**64
    tile_indices = torch.arange(d1 * d2, dtype=torch.int64).view(d1, d2)
    # encode each position in tile as a tuple of <= 8 unique bytes
    permuted_tile_indices = torch.zeros_like(tile_indices)
    for i in range(8):
        # select i-th byte, apply transformation and trace where each index ended up
        ith_dim_indices = torch.div(tile_indices, 256**i, rounding_mode="trunc") % 256
        sample_tile_i = (ith_dim_indices - 128).to(torch.int8).contiguous()
        assert torch.all(sample_tile_i.int() + 128 == ith_dim_indices), "int overflow"
        permuted_tile_i = transform_tile(sample_tile_i)
        ith_permuted_indices = permuted_tile_i.to(tile_indices.dtype) + 128
        permuted_tile_indices += ith_permuted_indices * (256**i)
        if d1 * d2 < 256**i:
            break  # if all indices fit in i bytes, stop early
    return permuted_tile_indices


def undo_layout(permuted_tensor: torch.Tensor, tile_indices: torch.LongTensor) -> torch.Tensor:
    """
    Undo a tiled permutation such as turing or ampere layout

    :param permuted_tensor: torch tensor in a permuted layout
    :param tile_indices: reverse transformation indices, from get_inverse_transform_indices
    :return: contiguous row-major tensor
    """
    (rows, cols), (tile_rows, tile_cols) = permuted_tensor.shape, tile_indices.shape
    assert rows % tile_rows == cols % tile_cols == 0, "tensor must contain a whole number of tiles"
    tensor = permuted_tensor.reshape(-1, tile_indices.numel()).t()
    outputs = torch.empty_like(tensor)  # note: not using .index_copy because it was slower on cuda
    outputs[tile_indices.flatten()] = tensor
    outputs = outputs.reshape(tile_rows, tile_cols, cols // tile_cols, rows // tile_rows)
    outputs = outputs.permute(3, 0, 2, 1)  # (rows // tile_rows, tile_rows), (cols // tile_cols, tile_cols)
    return outputs.reshape(rows, cols).contiguous()


# the rest of this file is just a patch to bitsandbytes that modifies Linear8bitLt and dependencies


class CustomLinear8bitLt(Linear8bitLt):
    def __init__(self, *args, memory_efficient_backward: bool = False, **kwargs):
        assert not memory_efficient_backward, "memory_efficient_backward is no longer used"
        super().__init__(*args, **kwargs)
        self.state = CustomMatmulLtState(**dataclasses.asdict(self.state))

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = custom_matmul8bitlt(x, self.weight, bias=self.bias, state=self.state)
        if not self.state.has_fp16_weights:
            if self.state.CB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
        return out


@dataclasses.dataclass(init=True)
class CustomMatmulLtState(MatmulLtState):
    tile_indices: Optional[torch.Tensor] = None

    def get_tile_size(self):
        assert self.formatB in (
            "col_turing",
            "col_ampere",
        ), f"please find this assert and manually enter tile size for {self.formatB}"
        return (8, 32) if self.formatB == "col_turing" else "col_ampere"


def custom_matmul8bitlt(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor = None,
    state: CustomMatmulLtState = None,
    threshold=0.0,
    bias=None,
):
    state = state or MatmulLtState()
    if threshold > 0.0:
        state.threshold = threshold
    return CustomMatMul8bitLt.apply(A, B, out, bias, state)


class CustomMatMul8bitLt(MatMul8bitLt):
    # forward is the same as in inference-only CxB
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None
        req_gradA, req_gradB, _, req_gradBias, _ = ctx.needs_input_grad
        CAt, subA = ctx.tensors
        SCAt, idx = ctx.tensor_states
        formatB = ctx.formatB
        state = ctx.state
        grad_A = grad_B = grad_bias = None

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # Cast grad_output to fp16
        if len(grad_output.shape) == 3:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        Cgrad, Cgradt, SCgrad, SCgradt, coo_tensor = F.double_quant(grad_output.to(torch.float16))
        if req_gradB:
            CxAt, SAt = F.transform(CAt, formatB, transpose=True)
            C32grad, Sgrad = F.transform(Cgradt, "col32", transpose=True)
            gradB32, SgradB32 = F.igemmlt(C32grad, CxAt, Sgrad, SAt)
            grad_B = F.mm_dequant(gradB32, SgradB32, SCgradt, SCAt)
            if state.threshold > 0.0 and subA is not None:
                grad_B[:, idx] += torch.matmul(grad_output.t(), subA)

        if req_gradA:
            if state.CBt is not None:
                C32grad, Sgrad = F.transform(Cgrad, "col32")
                if state.CxBt is None:
                    state.CxBt, state.SBt = F.transform(state.CBt, to_order=formatB, transpose=True)
                gradA32, SgradA32 = F.igemmlt(C32grad, state.CxBt, Sgrad, state.SBt)
                grad_A = F.mm_dequant(gradA32, SgradA32, SCgrad, state.SCBt).view(ctx.grad_shape).to(ctx.dtype_A)

            elif state.CB is not None:
                CB = state.CB.to(ctx.dtype_A, copy=True).mul_(state.SCB.unsqueeze(1).mul(1.0 / 127.0))
                grad_A = torch.matmul(grad_output, CB).view(ctx.grad_shape).to(ctx.dtype_A)
            elif state.CxB is not None:

                if state.tile_indices is None:
                    order, tile_size = state.formatB, state.get_tile_size()
                    transform = lambda x: F.transform(x.cuda(), from_order="row", to_order=order)[0].to(x.device)
                    with torch.no_grad():
                        state.tile_indices = get_inverse_transform_indices(transform, tile_size).to(state.CxB.device)

                CB = (
                    undo_layout(state.CxB, state.tile_indices)
                    .to(ctx.dtype_A)
                    .mul_(state.SCB.unsqueeze(1).mul(1.0 / 127.0))
                )
                grad_A = torch.matmul(grad_output, CB).view(ctx.grad_shape).to(ctx.dtype_A)
            else:
                raise Exception("State must contain either CBt or CB or CxB matrix for backward")

        return grad_A, grad_B, None, grad_bias, None
