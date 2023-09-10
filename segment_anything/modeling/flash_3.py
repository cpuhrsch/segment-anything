"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""

import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(1, 0), padding_option="zero")
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = P_SEQ + (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX + P_SEQ
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(1, 0), padding_option="zero")
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
        if IS_CAUSAL:
            qk = tl.where(P_SEQ + offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k, out_dtype=tl.float16)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    if start_m * BLOCK_M < N_CTX:
        tl.store(O_block_ptr, acc.to(tl.float16))


def attention(q, k, v, causal, sm_scale):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    BLOCK_M = 128
    BLOCK_N = 64 if Lk <= 64 else 32
    num_stages = 4 if Lk <= 64 else 3
    num_warps = 4
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    P_SEQ = 0 if q.shape[-2] == k.shape[-2] else k.shape[-2] - q.shape[-2]
    _fwd_kernel[grid](
        q, k, v, sm_scale,
        L,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2], P_SEQ,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=num_stages)

    return o



# attention = _attention.apply


# @pytest.mark.parametrize('Z, H, N_CTX, D_HEAD, P_SEQ', [(6, 9, 1024, 64, 128)])
# @pytest.mark.parametrize('causal', [False, True])
# def test_op(Z, H, N_CTX, D_HEAD, P_SEQ, causal, dtype=torch.float16):
#     torch.manual_seed(20)
#     q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
#     k = torch.empty((Z, H, N_CTX + P_SEQ, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
#     v = torch.empty((Z, H, N_CTX + P_SEQ, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
#     sm_scale = 0.5
#     dout = torch.randn_like(q)
#     # reference implementation
#     M = torch.tril(torch.ones((N_CTX, N_CTX + P_SEQ), device="cuda"), diagonal=P_SEQ)
#     p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
#     if causal:
#         p[:, :, M == 0] = float("-inf")
#     p = torch.softmax(p.float(), dim=-1).half()
#     # p = torch.exp(p)
#     ref_out = torch.matmul(p, v)
#     ref_out.backward(dout)
#     ref_dv, v.grad = v.grad.clone(), None
#     ref_dk, k.grad = k.grad.clone(), None
#     ref_dq, q.grad = q.grad.clone(), None
#     # triton implementation
#     tri_out = attention(q, k, v, causal, sm_scale).half()
#     tri_out.backward(dout)
#     tri_dv, v.grad = v.grad.clone(), None
#     tri_dk, k.grad = k.grad.clone(), None
#     tri_dq, q.grad = q.grad.clone(), None
#     # compare
#     assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
#     assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0)
#     assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0)
#     assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0)
# 
# 
# try:
#     from flash_attn.flash_attn_interface import \
#         flash_attn_qkvpacked_func as flash_attn_func
#     FLASH_VER = 2
# except BaseException:
#     try:
#         from flash_attn.flash_attn_interface import flash_attn_func
#         FLASH_VER = 1
#     except BaseException:
#         FLASH_VER = None
# HAS_FLASH = FLASH_VER is not None
# 
# BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# # vary seq length for fixed head and batch=4
# configs = [triton.testing.Benchmark(
#     x_names=['N_CTX'],
#     x_vals=[2**i for i in range(10, 15)],
#     line_arg='provider',
#     line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
#     line_names=['Triton'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []),
#     styles=[('red', '-'), ('blue', '-')],
#     ylabel='ms',
#     plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}',
#     args={'H': N_HEADS, 'BATCH': BATCH, 'D_HEAD': D_HEAD, 'dtype': torch.float16, 'mode': mode, 'causal': causal}
# ) for mode in ['fwd', 'bwd'] for causal in [False, True]]
# 
# 
# @triton.testing.perf_report(configs)
# def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
#     assert mode in ['fwd', 'bwd']
#     warmup = 25
#     rep = 100
#     if provider == "triton":
#         q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
#         k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
#         v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
#         sm_scale = 1.3
#         fn = lambda: attention(q, k, v, causal, sm_scale)
#         if mode == 'bwd':
#             o = fn()
#             do = torch.randn_like(o)
#             fn = lambda: o.backward(do, retain_graph=True)
#         ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
#     if provider == "flash":
#         qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
#         if FLASH_VER == 1:
#             lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
#             cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
#             cu_seqlens[1:] = lengths.cumsum(0)
#             qkv = qkv.reshape(BATCH * N_CTX, 3, H, D_HEAD)
#             fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=causal)
#         elif FLASH_VER == 2:
#             fn = lambda: flash_attn_func(qkv, causal=causal)
#         else:
#             raise ValueError(f'unknown {FLASH_VER = }')
#         if mode == 'bwd':
#             o = fn()
#             do = torch.randn_like(o)
#             fn = lambda: o.backward(do, retain_graph=True)
#         ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
#     flops_per_matmul = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD
#     total_flops = 2 * flops_per_matmul
#     if causal:
#         total_flops *= 0.5
#     if mode == 'bwd':
#         total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
#     return total_flops / ms * 1e-9
# 
# 
# # only works on post-Ampere GPUs right now
# bench_flash_attention.run(save_path='.', print_data=True)
# 
