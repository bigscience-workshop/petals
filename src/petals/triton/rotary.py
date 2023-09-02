import triton
import triton.language as tl
import torch


@triton.jit
def get_freq_multi_tokens(offs_cn, starting_idx, theta: tl.constexpr, NB_TOKENS: tl.constexpr):
    DIM: tl.constexpr = 128  # in model, dim = self.params.dim // self.params.n_heads
    freqs = offs_cn % DIM
    freqs = freqs.to(tl.float32) / DIM
    freqs = tl.math.pow(theta, freqs)
    freqs = (tl.arange(0, NB_TOKENS) + starting_idx)[:, None] / freqs[None, :]
    return tl.cos(freqs), tl.sin(freqs)


@triton.jit
def rbe_triton(
    x_ptr,
    out_ptr,
    M,
    K,
    stride_x_batch,
    stride_x_m,
    stride_x_n,
    stride_out_batch,
    stride_out_m,
    stride_out_n,
    start_token_position,
    THETA: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
    pid_n = pid % tl.cdiv(K, BLOCK_SIZE_K)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K // 2) * 2  # take only even numbers
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] + stride_x_n * offs_n[None, :])
    x_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
    x_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    imag = tl.load(x_ptrs + 1, mask=x_imag_mask, other=0.0)
    tl.debug_barrier()
    start_block = start_token_position + pid_m * BLOCK_SIZE_M
    cos, sin = get_freq_multi_tokens(offs_cn=offs_n, starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_M)

    out_real = real * cos - imag * sin
    out_imag = real * sin + imag * cos
    tl.debug_barrier()
    out_ptrs = out_ptr + (
        pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] + stride_out_n * offs_n[None, :]
    )
    out_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    tl.store(out_ptrs, out_real, mask=out_real_mask)
    out_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    tl.store(out_ptrs + 1, out_imag, mask=out_imag_mask)


def rbe_triton_wrapper(x: torch.Tensor, pos: int) -> torch.Tensor:
    batch, M, K = x.shape
    out = torch.empty_like(x)
    grid = lambda META: (
        batch,
        triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["K"], META["BLOCK_SIZE_K"]),
    )

    rbe_triton[grid](
        x,
        out,
        M,
        K,
        *x.stride(),
        *out.stride(),
        start_token_position=pos,
        THETA=10000.0,
        BLOCK_SIZE_M=2,
        BLOCK_SIZE_K=1024
    )
    return out
