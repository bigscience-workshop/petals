import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += tl.math.pow(x.to(tl.float32), 2)

    var = tl.sum(var, axis=0) / N_SIZE
    rstd = tl.math.rsqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)


def rmsnorm_triton_wrapper(x, rms_w, eps=1e-6):
    batch_size, seq_length, hid_dim = x.shape
    assert rms_w.shape[-1] == hid_dim
    out = torch.empty_like(x)
    rmsnorm_triton[(batch_size, seq_length,)](x, rms_w, out,
                                *x.stride(),
                                *rms_w.stride(),
                                *out.stride(),
                                N_SIZE=hid_dim, eps=eps, BLOCK_N_SIZE=1024,
                                )
    return out