import math

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    N_HEAD,
    H,
    N_CTX,
    start_position,  # <- ADDED
    IS_CAUSAL: tl.constexpr,  # <- ADDED
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    start_m = tl.program_id(0)

    head_idx = tl.program_id(1)
    batch_id = head_idx // N_HEAD
    off_hz = head_idx % N_HEAD

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = (
        batch_id * stride_qz + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    )  # <- stride fixed
    off_k = (
        batch_id * stride_kz + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    )  # <- stride fixed
    off_v = (
        batch_id * stride_vz + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    )  # <- stride fixed
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, offs_m[:, None] < H, other=0.0)
    # loop over k, v and update accumulator
    block_n_end = N_CTX  # <- ADDED (including the IF)
    if IS_CAUSAL:
        # in causal mode, we expect that BLOCK_M_SIZE == BLOCK_N_SIZE
        # autotune will prune shapes not matching this rule
        block_n_end = (start_m + 1) * BLOCK_N + start_position
    for start_n in range(0, block_n_end, BLOCK_N):
        block_n_offs = start_n + offs_n  # <- ADDED
        # -- compute qk ----
        k = tl.load(k_ptrs, block_n_offs[:, None] < N_CTX, 0.0)
        if USE_FP8:
            k = k.to(tl.float8e5, bitcast=True)
            k = k.to(tl.float16)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk = tl.where(offs_n[None, :] < N_CTX, qk, float("-inf"))  # <- ADDED
        qk *= sm_scale
        if IS_CAUSAL:  # <- ADDED
            qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + start_position), qk, float("-inf"))

        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1.0 / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs, block_n_offs[:, None] < N_CTX, 0.0)
        if USE_FP8:
            v = v.to(tl.float8e5, bitcast=True)
            v = v.to(tl.float16)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_o = batch_id * stride_oz + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, offs_m[:, None] < H)


def triton_fa(q, k, v, sm_scale, is_causal, start_position):
    assert q.dtype == torch.float16
    assert k.dtype == v.dtype and k.dtype in [torch.float16, torch.int8]

    BLOCK = 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    num_warps = 4 if Lk <= 64 else 8
    batch, head_size, m_size, dhead = q.size()
    grid = (triton.cdiv(m_size, BLOCK), head_size * batch)
    n_size = k.size(2)
    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        head_size,
        m_size,
        n_size,
        start_position=start_position,
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=Lk,
        USE_FP8=k.dtype == torch.int8,  # USE_FP8
        num_warps=num_warps,
        num_stages=2,
    )

    return o


def attention_triton_wrapper(q, k, v, head_dim):
    return triton_fa(q, k, v, sm_scale=1 / math.sqrt(head_dim), is_causal=True, start_position=0)
