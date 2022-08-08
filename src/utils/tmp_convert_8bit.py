# This file contains a temporary solution for int8 backward until the new 'bitsandbytes' release will come.
# To be removed

import operator
from functools import reduce  # Required in Python 3

import bitsandbytes as bnb
import bitsandbytes.functional as F
import torch


def replace_8bit_linear(model, threshold=6.0):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules from the `bitsandbytes`
    library. This will enable running your models using mixed int8 precision as described by the paper `GPT3.int8():
    8-bit Matrix Multiplication for Transformers at Scale`. Make sure `bitsandbytes` compiled with the correct CUDA
    version of your hardware is installed before running this function. `pip install -i https://test.pypi.org/simple/
    bitsandbytes-cudaXXX` with `XXX` is your CUDA version (e.g., 11.6 = 116)
    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function.
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

        if isinstance(module, torch.nn.Linear) and n != ["lm_head", "score"]:
            model._modules[n] = Linear8bitLtFor8bitBackward(
                module.in_features,
                module.out_features,
                module.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
            ).to(model._modules[n].weight.device)
    return model


class Linear8bitLtFor8bitBackward(bnb.nn.Linear8bitLt):
    def forward(self, x):
        self.state.is_training = self.training

        if self.weight.CB is not None:
            self.init_8bit_state()

        out = matmul(x, self.weight, state=self.state)

        if self.bias is not None:
            out += self.bias.unsqueeze(0).expand_as(out)

        if not self.state.has_fp16_weights and self.state.CB is not None:
            # we converted 8-bit row major to turing/ampere format in the first inference pass
            # we no longer need the row-major weight
            del self.state.CB
            self.weight.data = self.state.CxB

        return out


# math.prod not compatible with python < 3.8
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class MatMul8bitLt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, out=None, state=bnb.autograd._functions.MatmulLtState()):
        # default to pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            if A.shape[-1] == B.shape[0]:
                return torch.empty(A.shape[:-1] + B.shape[1:], dtype=torch.float16, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B.shape[:1], dtype=torch.float16, device=A.device)

        # 1. Quantize A
        # 2. Quantize B
        # 3. Matmul
        # 4. Mixed-precision decomposition matmul
        # 5. Save state
        requires_gradA = A.requires_grad
        requires_gradB = B.requires_grad
        formatB = state.formatB
        input_shape = A.shape
        if state.outlier_pool is None:
            state.outlier_pool = bnb.autograd._functions.GlobalOutlierPooler.get_instance()
        assert A.dtype == torch.float16, f"The input data type needs to be fp16 but {A.dtype} was found!"

        # 1. Quantize A
        if len(A.shape) == 3:
            A = A.view(-1, A.shape[-1]).contiguous()
        CA, CAt, SCA, SCAt, coo_tensorA = F.double_quant(A, threshold=state.threshold)

        if state.threshold > 0.0 and coo_tensorA is not None:
            if state.has_fp16_weights:
                idx = torch.unique(coo_tensorA.colidx).long()
                CA[:, idx] = 0
                CAt[:, idx] = 0
                subA = A[:, idx]
                state.subB = B[:, idx].t().contiguous()
                state.idx = idx
            else:
                if state.CxB is None:
                    # B in in 8-bit row-major, we can transform it back to 16-bit to extract outlier dimensions
                    # we also need to convert it to the turing/ampere format
                    state.CxB, state.SB = F.transform(state.CB, to_order=formatB)
        else:
            if not state.has_fp16_weights and state.CxB is None:
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
                ) = F.double_quant(B)
                state.CxB, state.SB = F.transform(CB, to_order=formatB)
        else:
            has_grad = False

        if coo_tensorA is not None and not state.has_fp16_weights:
            # extract outliers

            outlier_idx = torch.unique(coo_tensorA.colidx)
            state.idx = outlier_idx
            outliers = F.extract_outliers(state.CxB, state.SB, state.idx.int())
            state.subB = (outliers * state.SCB.view(-1, 1) / 127.0).t().contiguous().half()
            CA[:, state.idx.long()] = 0
            CAt[:, state.idx.long()] = 0
            subA = A[:, state.idx.long()]

        shapeB = state.SB[0]

        if len(input_shape) == 3:
            output_shape = (input_shape[0], input_shape[1], shapeB[0])
        else:
            output_shape = (input_shape[0], shapeB[0])

        # 3. Matmul
        C32A, SA = F.transform(CA, "col32")
        out32, Sout32 = F.igemmlt(C32A, state.CxB, SA, state.SB)
        output = F.mm_dequant(out32, Sout32, SCA, state.SCB)

        # 4. Mixed-precision decomposition matmul
        if coo_tensorA is not None and subA is not None:
            output += torch.matmul(subA, state.subB)

        # 5. Save state
        ctx.state = state

        ctx.formatB = formatB
        ctx.grad_shape = input_shape
        ctx.req_grads = [requires_gradA, requires_gradB]

        if requires_gradA or requires_gradB:
            ctx.tensors = (CAt, subA)
            ctx.tensor_states = (SCAt, state.idx)
        else:
            ctx.tensors = [None, None]
            ctx.tensor_states = (None, None)
            ctx.save_for_backward(None, None)

        # clone_func = torch.clone if len(output_shape) == 3 else lambda x : x
        clone_func = torch.clone
        return clone_func(output.view(output_shape))

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, None
        req_gradA, req_gradB = ctx.req_grads
        CAt, subA = ctx.tensors
        SCAt, idx = ctx.tensor_states
        formatB = ctx.formatB
        state = ctx.state

        if len(grad_output.shape) == 3:
            grad_output = grad_output.view(-1, grad_output.shape[-1]).contiguous()

        grad_A = grad_B = None

        Cgrad, Cgradt, SCgrad, SCgradt, coo_tensor = F.double_quant(grad_output)
        if req_gradB:
            CxAt, SAt = F.transform(CAt, formatB, transpose=True)
            C32grad, Sgrad = F.transform(Cgradt, "col32", transpose=True)
            gradB32, SgradB32 = F.igemmlt(C32grad, CxAt, Sgrad, SAt)
            grad_B = F.mm_dequant(gradB32, SgradB32, SCgradt, SCAt)
            if state.threshold > 0.0 and subA is not None:
                grad_B[:, idx] += torch.matmul(grad_output.t(), subA)

        if req_gradA:
            C32grad, Sgrad = F.transform(Cgrad, "col32")
            if state.CxBt is None:
                state.CxBt, state.SBt = F.transform(state.CBt, to_order=formatB, transpose=True)
            gradA32, SgradA32 = F.igemmlt(C32grad, state.CxBt, Sgrad, state.SBt)
            grad_A = F.mm_dequant(gradA32, SgradA32, SCgrad, state.SCBt).view(ctx.grad_shape)

        return grad_A, grad_B, None, None


def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor = None,
    state: bnb.autograd._functions.MatmulLtState = None,
    threshold=0.0,
):
    state = state or bnb.autograd._functions.MatmulLtState()
    if threshold > 0.0:
        state.threshold = threshold
    return MatMul8bitLt.apply(A, B, out, state)
