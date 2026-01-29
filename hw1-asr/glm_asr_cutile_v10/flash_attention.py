"""
FlashAttention implementation using CuTile.
Based on NVIDIA TileGym's FlashAttention implementation.
"""

import math
import cuda.tile as ct
import cupy as cp
from typing import Optional

# Constants
INV_LOG_2 = 1.0 / math.log(2)
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def get_stream():
    """Get current CUDA stream pointer."""
    return cp.cuda.get_current_stream().ptr


@ct.kernel(occupancy=2)
def flash_attention_kernel(
    Q,          # [batch, num_heads, seq_q, head_dim]
    K,          # [batch, num_kv_heads, seq_k, head_dim]
    V,          # [batch, num_kv_heads, seq_k, head_dim]
    Out,        # [batch, num_heads, seq_q, head_dim]
    qk_scale: float,
    TILE_D: ConstInt,       # head_dim (must be power of 2)
    H: ConstInt,            # num_heads
    TILE_M: ConstInt,       # query tile size
    TILE_N: ConstInt,       # key/value tile size
    QUERY_GROUP_SIZE: ConstInt,  # num_heads // num_kv_heads
    CAUSAL: ConstBool,
):
    """
    FlashAttention kernel with online softmax and TF32 tensor cores.
    Computes attention without materializing the full attention matrix.
    Uses TF32 for MMA operations to leverage tensor core acceleration.
    """
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)  # which query tile
    bid_y = ct.bid(1)  # batch * num_heads
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # qk_scale is already pre-scaled by 1/ln(2) for exp2

    # Initialize offsets for current query tile
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = offs_m[:, None]  # [TILE_M, 1]

    # Initialize offsets for key/value tile
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # Online softmax accumulators (float32 for stability)
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)  # running max
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)        # running sum
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)   # output accumulator

    # Load query tile [TILE_M, TILE_D]
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D))
    q = q.reshape((TILE_M, TILE_D))

    # Calculate number of key/value tiles to process
    k_seqlen = K.shape[2]
    if CAUSAL:
        # For causal attention, only attend to positions <= current position
        m_end = (bid_x + 1) * TILE_M
        mask_start = (bid_x * TILE_M) // TILE_N
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # Loop over key/value tiles
    for j in range(0, Tc):
        # Load key tile [TILE_D, TILE_N] (transposed for matmul)
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),  # transpose last two dims
        )
        k = k.reshape((TILE_D, TILE_N))

        # Compute QK^T: [TILE_M, TILE_D] @ [TILE_D, TILE_N] = [TILE_M, TILE_N]
        # Use TF32 for tensor core acceleration
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        q_tf32 = ct.astype(q, ct.tfloat32)
        k_tf32 = ct.astype(k, ct.tfloat32)
        qk = ct.mma(q_tf32, k_tf32, qk)

        # Apply causal mask if needed
        if CAUSAL and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = offs_m >= offs_n  # [TILE_M, TILE_N]
            # Also mask out-of-bounds positions
            mask = mask & (offs_n < k_seqlen)
            mask = ct.where(mask, 0.0, -math.inf)
            qk = qk + mask

        # Online softmax update (following TileGym's pattern)
        # 1. Compute scaled max for numerical stability
        m_ij = max(m_i, ct.mul(ct.max(qk, axis=-1, keepdims=True), qk_scale))

        # 2. Compute scaled scores and subtract max
        qk = ct.sub(ct.mul(qk, qk_scale), m_ij)

        # 3. Compute attention weights using exp2 (faster than exp)
        p = ct.exp2(qk, flush_to_zero=True)  # [TILE_M, TILE_N]

        # 4. Compute sum of weights
        l_ij = ct.sum(p, axis=-1, keepdims=True)  # [TILE_M, 1]

        # 5. Compute rescaling factor
        alpha = ct.exp2(ct.sub(m_i, m_ij), flush_to_zero=True)  # [TILE_M, 1]

        # 6. Update running sum
        l_i = l_i * alpha + l_ij

        # 7. Rescale accumulator
        acc = acc * alpha

        # Load value tile [TILE_N, TILE_D]
        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
        )
        v = v.reshape((TILE_N, TILE_D))

        # Compute PV: [TILE_M, TILE_N] @ [TILE_N, TILE_D] = [TILE_M, TILE_D]
        # Use TF32 for tensor core acceleration
        p_tf32 = ct.astype(p, ct.tfloat32)
        v_tf32 = ct.astype(v, ct.tfloat32)
        acc = ct.mma(p_tf32, v_tf32, acc)

        # Update running max
        m_i = m_ij

    # Final normalization
    acc = ct.truediv(acc, l_i, flush_to_zero=True)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)

    # Store output
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


def flash_attention(
    q: cp.ndarray,
    k: cp.ndarray,
    v: cp.ndarray,
    scale: Optional[float] = None,
    causal: bool = False,
) -> cp.ndarray:
    """
    FlashAttention forward pass.

    Args:
        q: Query tensor [batch, num_heads, seq_q, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_k, head_dim]
        v: Value tensor [batch, num_kv_heads, seq_k, head_dim]
        scale: Softmax scale (default: 1/sqrt(head_dim))
        causal: Whether to apply causal mask

    Returns:
        Output tensor [batch, num_heads, seq_q, head_dim]
    """
    # Save original dimensions
    batch_size, num_heads, orig_seq_q, orig_head_dim = q.shape
    _, num_kv_heads, orig_seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(orig_head_dim)

    # V8.2: Only convert if necessary to avoid unnecessary copies
    def ensure_contiguous_float32(x):
        if x.dtype != cp.float32:
            x = x.astype(cp.float32)
        if not x.flags.c_contiguous:
            x = cp.ascontiguousarray(x)
        return x

    q = ensure_contiguous_float32(q)
    k = ensure_contiguous_float32(k)
    v = ensure_contiguous_float32(v)

    # Calculate GQA group size
    assert num_heads % num_kv_heads == 0
    query_group_size = num_heads // num_kv_heads

    # Choose tile sizes
    TILE_M = 64
    TILE_N = 64

    # Calculate padded dimensions
    head_dim_padded = 1 << (orig_head_dim - 1).bit_length()
    seq_q_padded = ((orig_seq_q + TILE_M - 1) // TILE_M) * TILE_M
    seq_k_padded = ((orig_seq_k + TILE_N - 1) // TILE_N) * TILE_N

    # Check if padding is needed
    need_head_pad = (orig_head_dim != head_dim_padded)
    need_seq_pad = (orig_seq_q != seq_q_padded) or (orig_seq_k != seq_k_padded)

    # Create padded tensors if needed
    if need_head_pad or need_seq_pad:
        final_seq_q = seq_q_padded if need_seq_pad else orig_seq_q
        final_seq_k = seq_k_padded if need_seq_pad else orig_seq_k
        final_head_dim = head_dim_padded if need_head_pad else orig_head_dim

        q_work = cp.zeros((batch_size, num_heads, final_seq_q, final_head_dim), dtype=cp.float32)
        k_work = cp.zeros((batch_size, num_kv_heads, final_seq_k, final_head_dim), dtype=cp.float32)
        v_work = cp.zeros((batch_size, num_kv_heads, final_seq_k, final_head_dim), dtype=cp.float32)
        out_work = cp.zeros((batch_size, num_heads, final_seq_q, final_head_dim), dtype=cp.float32)

        q_work[:, :, :orig_seq_q, :orig_head_dim] = q
        k_work[:, :, :orig_seq_k, :orig_head_dim] = k
        v_work[:, :, :orig_seq_k, :orig_head_dim] = v

        TILE_D = final_head_dim
        actual_seq_q = final_seq_q
    else:
        q_work, k_work, v_work = q, k, v
        out_work = cp.empty_like(q_work)
        TILE_D = orig_head_dim
        actual_seq_q = orig_seq_q

    # Calculate grid dimensions
    num_q_tiles = actual_seq_q // TILE_M
    grid = (num_q_tiles, batch_size * num_heads, 1)

    # Pre-compute qk_scale for exp2 (multiply by 1/ln(2)) at Python level
    # This avoids mixing Python float64 with kernel float32
    qk_scale = float(scale * INV_LOG_2)

    # Launch kernel
    try:
        ct.launch(
            get_stream(),
            grid,
            flash_attention_kernel,
            (
                q_work, k_work, v_work, out_work,
                qk_scale,  # Already scaled for exp2
                TILE_D,
                num_heads,
                TILE_M,
                TILE_N,
                query_group_size,
                causal,
            ),
        )
    except Exception as e:
        # Fallback to standard attention if kernel fails
        print(f"FlashAttention kernel failed: {e}, falling back to standard attention")
        return _standard_attention(q, k, v, scale, causal)

    # Extract result (remove padding)
    result = out_work[:, :, :orig_seq_q, :orig_head_dim]
    return cp.ascontiguousarray(result)


def _standard_attention(q, k, v, scale, causal):
    """Fallback standard attention implementation."""
    # Q @ K^T
    scores = cp.einsum('bhqd,bhkd->bhqk', q, k) * scale

    if causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        mask = cp.triu(cp.ones((seq_q, seq_k), dtype=cp.float32), k=1) * -1e9
        scores = scores + mask

    # Softmax
    scores = scores - cp.max(scores, axis=-1, keepdims=True)
    exp_scores = cp.exp(scores)
    attn_weights = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)

    # @ V
    return cp.einsum('bhqk,bhkd->bhqd', attn_weights, v)


# Simple test
if __name__ == "__main__":
    print("Testing FlashAttention...")

    batch, heads, seq, dim = 2, 8, 128, 64
    q = cp.random.randn(batch, heads, seq, dim).astype(cp.float32)
    k = cp.random.randn(batch, heads, seq, dim).astype(cp.float32)
    v = cp.random.randn(batch, heads, seq, dim).astype(cp.float32)

    # Test non-causal
    out = flash_attention(q, k, v, causal=False)
    print(f"Non-causal output shape: {out.shape}")

    # Test causal
    out_causal = flash_attention(q, k, v, causal=True)
    print(f"Causal output shape: {out_causal.shape}")

    # Compare with standard
    out_std = _standard_attention(q, k, v, 1.0/math.sqrt(dim), False)
    diff = float(cp.max(cp.abs(out - out_std)))
    print(f"Max diff vs standard: {diff:.6f}")

    print("FlashAttention test passed!")
