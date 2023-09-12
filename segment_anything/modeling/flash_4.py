"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""

import torch

import triton
import triton.language as tl


class _WipFlash2Library:
    lib = torch.library.Library("wipflash2", "DEF")
    ops_table: dict[tuple[str, str], callable] = {}

    @classmethod
    def registerOp(cls, op_key, full_schema, op_impl, dispatch_key):
        print("cls.ops_table: ", cls.ops_table)
        if (op_key, dispatch_key) not in cls.ops_table:
            if (op_key, "CUDA") not in cls.ops_table:
                cls.lib.define(full_schema)
            cls.lib.impl("wipflash2::" + op_key, op_impl, dispatch_key)
            cls.ops_table[(op_key, dispatch_key)] = op_impl
        return cls.ops_table[(op_key, dispatch_key)]


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def _fwd_kernel(
    Q, K, V, B0, sm_scale,
    Out,
    stride_qh, stride_qm,
    stride_kh, stride_kn,
    stride_vh, stride_vk,
    stride_oh, stride_om,
    stride_b0h, stride_b0m,
    Z,
    H,
    N_CTX,
    P_SEQ,
    BIAS_LAST_SIZE: tl.constexpr,
    B0_NUMEL: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
#    **META):
     BLOCK_M: tl.constexpr,
     BLOCK_N: tl.constexpr,
):
    # BLOCK_M = META['BLOCK_M']
    # BLOCK_N = META['BLOCK_N']
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(1, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    # initialize offsets
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr) #, boundary_check=(1, 0), padding_option="zero")
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = N_CTX + P_SEQ

    b_mask = tl.arange(0, BLOCK_N)
    b_ptr_offsets_m = tl.arange(0, BLOCK_M)
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr) #, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr) #, boundary_check=(1, 0), padding_option="zero")
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
        qk += tl.dot(q, k, out_dtype=tl.float16) # * qk_scale).to(tl.float16)

        # -- compute rel_h[:, None] + rel_w[None, :] bias ---

        # Bias

        # Get to the right batch + head
        b_offset = off_hz * stride_b0h

        # # Get to the right rows
        # b_ptr_offsets_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # b_ptr_offsets_m = b_ptr_offsets_m * stride_b0m

        # Get to the right column subsection
        # bias_last_size = (B0_NUMEL - 4) // 2
        b_ptr_offsets_n_0 = (start_n + b_mask) // BIAS_LAST_SIZE
        b_ptr_offsets_n_1 = ((start_n + b_mask) % BIAS_LAST_SIZE) + BIAS_LAST_SIZE
        # b_ptr_offsets_n_0 = b_ptr_offsets_n_0 * stride_b0n
        # b_ptr_offsets_n_1 = b_ptr_offsets_n_1 * stride_b0n

        # # Construct the block of pointers
        # b_ptr_offsets_0 = b_ptr_offsets_m + b_ptr_offsets_n_0[None, :]
        # b_ptr_offsets_1 = b_ptr_offsets_m + b_ptr_offsets_n_1[None, :]

        # Combine and load
        # b_mask = (start_n + tl.arange(0, BLOCK_N)) < ((B0_NUMEL - 4) * (B0_NUMEL - 4))
        # qk += tl.load(B0 + b_offset + b_ptr_offsets_m[:, None] + b_ptr_offsets_n_0[None, :], eviction_policy='evict_last', mask=b_mask[None, :], other=float('-inf'))
        # qk += tl.load(B0 + b_offset + b_ptr_offsets_m[:, None] + b_ptr_offsets_n_1[None, :], eviction_policy='evict_last', mask=b_mask[None, :], other=float('-inf'))
        qk += tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m) * stride_b0m)[:, None] + b_ptr_offsets_n_0[None, :], mask=((start_n + b_mask) < ((B0_NUMEL - 4) * (B0_NUMEL - 4)))[None, :], other=float('-inf'))
        qk += tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m) * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :], mask=((start_n + b_mask) < ((B0_NUMEL - 4) * (B0_NUMEL - 4)))[None, :], other=float('-inf'))

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

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))

def _attention_rel_h_rel_w_kernel(q, k, v, rel_h_w, sm_scale):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)

    BLOCK_M = 128

    BLOCK_N = 64 if Lk <= 64 else 32
    num_stages = 4 if Lk <= 64 else 3

    BLOCK_N = 64
    num_stages = 4

    num_warps = 4

    # BLOCK_M = 32 # 128
    # BLOCK_N = 32
    # num_warps = 8
    # num_stages = 8

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    # print("q.shape[0] * q.shape[1]: ", q.shape[0] * q.shape[1])
    P_SEQ = 0 if q.shape[-2] == k.shape[-2] else k.shape[-2] - q.shape[-2]
    assert P_SEQ == 0
    # assert rel_h.stride(0) == rel_w.stride(0)
    # assert rel_h.stride(1) == rel_w.stride(1)
    # assert rel_h.stride(2) == rel_w.stride(2)
    # assert rel_h.stride(3) == rel_w.stride(3)
    # assert rel_h.size(-1)  == rel_w.size(-1)
    b = rel_h_w
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert b.is_contiguous()
    _fwd_kernel[grid](
        q, k, v,
        b,
        sm_scale,
        o,
        q.stride(1), q.stride(2),
        k.stride(1), k.stride(2),
        v.stride(1), v.stride(2),
        o.stride(1), o.stride(2),
        b.stride(1), b.stride(2),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        P_SEQ,
        BIAS_LAST_SIZE=((b.size(-1) - 4) // 2),
        B0_NUMEL=b.size(-1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=Lk,
        num_warps=num_warps,
        num_stages=num_stages)

    return o

def _attention_rel_h_rel_w(q_, k_, v_, rel_h_, rel_w_):
    """
    Implements SDPA but bias is addition of (rel_h + rel_w).view(..., rel_h.size(-2) * rel_w.size(-1))
    """

    import math
    sm_scale = 1. / math.sqrt(q_.size(-1))
    q_size_2_padded = (((q_.size(-2) + 256 - 1) // 256) * 256) - q_.size(-2)
    q = torch.nn.functional.pad(q_, (0, 0, 0, q_size_2_padded), "constant", 0).contiguous()
    k = torch.nn.functional.pad(k_, (0, 0, 0, q_size_2_padded), "constant", 0).contiguous()
    v = torch.nn.functional.pad(v_, (0, 0, 0, q_size_2_padded), "constant", 0).contiguous()

    # rel_h = torch.nn.functional.pad(rel_h_.squeeze(-1), (0, 2, 0, q_size_2_padded), "constant", float("-inf"))
    # rel_w = torch.nn.functional.pad(rel_w_.squeeze(-2), (0, 2, 0, q_size_2_padded), "constant", float("-inf"))
    rel_h_w_ = torch.cat([rel_h_.squeeze(-1), rel_w_.squeeze(-2)], dim=-1)
    rel_h_w = torch.nn.functional.pad(rel_h_w_, (0, 4, 0, q_size_2_padded), "constant", float("-inf"))

    # o = _attention_rel_h_rel_w_kernel(q, k, v, rel_h_w, sm_scale)
    o = torch.ops.wipflash2.mah_flash(q, k, v, rel_h_w, sm_scale)
    return o[:, :, :q_.size(-2), :].contiguous()


_WipFlash2Library.registerOp(
    "mah_flash",
    "mah_flash(Tensor q, Tensor k, Tensor v, Tensor rel_h_w, float sm_scale) -> Tensor",
    _attention_rel_h_rel_w_kernel,
    "CUDA",
)


def _attention_rel_h_rel_w_kernel_meta(q_, k_, v_, rel_h_w, sm_scale):
    return q_


_WipFlash2Library.registerOp(
    "mah_flash",
    "mah_flash(Tensor q, Tensor k, Tensor v, Tensor rel_h_w, float sm_scale) -> Tensor",
    _attention_rel_h_rel_w_kernel_meta,
    "Meta",
)
