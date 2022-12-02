"""
A patch to bitsandbytes 0.34.0 that introduces an option to run backward pass in default (fast) matrix layout.
Authors: modification by @borzunov, original code by @timdettmers. Please disregard commit authors in this file.

Core idea: layouts apply the same permutation to every tile in the matrix. We can treat this as (batched) gather ops.
  Reshape input tensor so that ij-th gather operation op will apply to ij-th elements in each tile.
Prototype: https://colab.research.google.com/drive/1EJ0MKifajXSSVq7O2_QGwtb0l6gRAGrh?usp=sharing
Based on: https://github.com/TimDettmers/bitsandbytes/blob/main/csrc/kernels.cu#L2130-L2136
Exact match tests: see $REPO/tests/test_linear8bitlt.py
"""
import dataclasses
import logging
from typing import Optional, Tuple

import bitsandbytes.functional as F
import torch
from bitsandbytes.autograd._functions import GlobalOutlierPooler, MatMul8bitLt, MatmulLtState, prod
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
        old_state, self.state = self.state, CustomMatmulLtState()
        self.state.threshold = old_state.threshold
        self.state.has_fp16_weights = old_state.has_fp16_weights
        self.state.memory_efficient_backward = old_state.memory_efficient_backward
        if old_state.threshold > 0.0 and not old_state.has_fp16_weights:
            self.state.use_pool = True

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = custom_matmul8bitlt(x, self.weight, bias=self.bias, state=self.state)
        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
        return out


@dataclasses.dataclass(init=True)
class CustomMatmulLtState(MatmulLtState):
    tile_indices: Optional[torch.Tensor] = None
    force_no_igemmlt: bool = False

    def get_tile_size(self):
        assert self.formatB in (
            "col_turing",
            "col_ampere",
        ), f"please find this assert and manually enter tile size for {self.formatB}"
        return (8, 32) if self.formatB == "col_turing" else (32, 32)


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
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, state=CustomMatmulLtState):
        using_igemmlt = torch.cuda.get_device_capability(device=A.device) >= (7, 5) and not state.force_no_igemmlt
        # default to pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            if A.shape[-1] == B.shape[0]:
                return torch.empty(A.shape[:-1] + B.shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B.shape[:1], dtype=A.dtype, device=A.device)

        # 1. Quantize A
        # 2. Quantize B
        # 3. Matmul
        # 4. Mixed-precision decomposition matmul
        # 5. Save state
        formatB = state.formatB
        input_shape = A.shape
        if state.outlier_pool is None:
            state.outlier_pool = GlobalOutlierPooler.get_instance()

        # Cast A to fp16
        if A.dtype != torch.float16:
            logging.debug(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")

        # 1. Quantize A
        if len(A.shape) == 3:
            A = A.view(-1, A.shape[-1]).contiguous()
        CA, CAt, SCA, SCAt, coo_tensorA = F.double_quant(A.to(torch.float16), threshold=state.threshold)

        if state.threshold > 0.0 and coo_tensorA is not None:
            if state.has_fp16_weights:
                idx = torch.unique(coo_tensorA.colidx).long()
                CA[:, idx] = 0
                CAt[:, idx] = 0
                subA = A[:, idx]
                state.subB = B[:, idx].t().contiguous()
                state.idx = idx
            else:
                if state.CxB is None and using_igemmlt:
                    # B in in 8-bit row-major, we can transform it back to 16-bit to extract outlier dimensions
                    # we also need to convert it to the turing/ampere format
                    state.CxB, state.SB = F.transform(state.CB, to_order=formatB)
        else:
            if not state.has_fp16_weights and state.CxB is None and using_igemmlt:
                state.CxB, state.SB = F.transform(state.CB, to_order=formatB)
            subA = None

        # 2. Quantize B
        if state.has_fp16_weights:
            has_grad = True if (getattr(B, "grad", None) is not None) else False
            is_transposed = not B.is_contiguous() and B.shape[0] == B.stride(1)
            if is_transposed:
                B = B.contiguous()

            if (state.is_training and not has_grad) or state.CxB is None:
                state.reset_grads()
                (
                    CB,
                    state.CBt,
                    state.SCB,
                    state.SCBt,
                    coo_tensorB,
                ) = F.double_quant(B.to(torch.float16))
                if using_igemmlt:
                    state.CxB, state.SB = F.transform(CB, to_order=formatB)
                else:
                    state.CB = CB
        else:
            has_grad = False

        if coo_tensorA is not None and not state.has_fp16_weights:
            # extract outliers

            outlier_idx = torch.unique(coo_tensorA.colidx)
            state.idx = outlier_idx
            # state.outlier_pool.add_outliers(outlier_idx, A.shape[-1])
            # if state.use_pool and state.outlier_pool.model_dim == A.shape[-1]:
            #    # do not use pool for 2nd FFN layer
            #    state.idx = state.outlier_pool.get_current_outlier_idx().to(A.device)
            # else:
            #    state.idx = outlier_idx
            if state.CxB is not None:
                outliers = F.extract_outliers(state.CxB, state.SB, state.idx.int())
            else:
                outliers = state.CB[:, state.idx.long()].clone()

            state.subB = (outliers * state.SCB.view(-1, 1) / 127.0).t().contiguous().to(A.dtype)
            CA[:, state.idx.long()] = 0
            CAt[:, state.idx.long()] = 0
            subA = A[:, state.idx.long()]

        shapeB = state.SB[0] if state.SB else B.shape

        if len(input_shape) == 3:
            output_shape = (input_shape[0], input_shape[1], shapeB[0])
        else:
            output_shape = (input_shape[0], shapeB[0])

        # 3. Matmul
        if using_igemmlt:
            C32A, SA = F.transform(CA, "col32")
            out32, Sout32 = F.igemmlt(C32A, state.CxB, SA, state.SB)
            if bias is None or bias.dtype == torch.float16:
                # we apply the fused bias here
                output = F.mm_dequant(out32, Sout32, SCA, state.SCB, bias=bias)
                output = output.to(A.dtype)
            else:  # apply bias separately
                output = F.mm_dequant(out32, Sout32, SCA, state.SCB, bias=None)
                output = output.to(A.dtype).add_(bias)

        else:
            A_wo_outliers = A.clone()
            if state.idx is not None:
                A_wo_outliers[:, state.idx.long()] = 0
            output = torch.nn.functional.linear(A_wo_outliers, state.CB.to(A.dtype))
            output = output.mul_(state.SCB.unsqueeze(0).mul(1.0 / 127.0))
            if bias is not None:
                output = output.add_(bias)

        # 4. Mixed-precision decomposition matmul
        if coo_tensorA is not None and subA is not None:
            output += torch.matmul(subA, state.subB)

        # 5. Save state
        ctx.state = state

        ctx.formatB = formatB
        ctx.grad_shape = input_shape
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (CAt, subA)
            ctx.tensor_states = (SCAt, state.idx)
        else:
            ctx.tensors = [None, None]
            ctx.tensor_states = (None, None)
            ctx.save_for_backward(None, None)

        clone_func = torch.clone if len(output_shape) == 3 else lambda x: x
        return clone_func(output.view(output_shape))

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
