"""
Optimized CuTile attention for decode mode (seq_q=1).
Uses tensor cores and avoids unnecessary padding overhead.
"""

import cuda.tile as ct
import cupy as cp
import math
from typing import Optional


def get_stream():
    """Get current CUDA stream pointer."""
    return cp.cuda.get_current_stream().ptr


@ct.kernel(occupancy=4)
def decode_attention_kernel(
    Q,          # [batch*heads, 1, head_dim] - single query per head
    K,          # [batch*kv_heads, seq_k, head_dim]
    V,          # [batch*kv_heads, seq_k, head_dim]
    Out,        # [batch*heads, 1, head_dim]
    qk_scale: float,
    head_dim: ct.Constant[int],
    seq_k: ct.Constant[int],
    TILE_K: ct.Constant[int],
    QUERY_GROUP_SIZE: ct.Constant[int],
):
    """
    Fused decode attention kernel for seq_q=1.
    Computes: softmax(Q @ K^T / sqrt(d)) @ V

    Each block handles one (batch, head) pair.
    Uses online softmax to avoid materializing full attention scores.
    """
    bid = ct.bid(0)  # batch * num_heads
    kv_head = bid // QUERY_GROUP_SIZE

    # Load query vector [1, 1, head_dim] - match 3D input
    q_3d = ct.load(Q, index=(bid, 0, 0), shape=(1, 1, head_dim))
    q = ct.reshape(q_3d, (1, head_dim))  # Reshape to 2D for matmul

    # Online softmax state
    m_prev = ct.full((1, 1), -math.inf, dtype=ct.float32)  # running max
    l_prev = ct.full((1, 1), 0.0, dtype=ct.float32)        # running sum
    acc = ct.zeros((1, head_dim), dtype=ct.float32)        # output accumulator

    num_k_tiles = ct.cdiv(seq_k, TILE_K)

    for k_idx in range(num_k_tiles):
        # Load K tile [1, TILE_K, head_dim] - index must be k_idx * TILE_K
        k_start = k_idx * TILE_K
        k_3d = ct.load(K, index=(kv_head, k_start, 0), shape=(1, TILE_K, head_dim), latency=3)
        k_tile = ct.reshape(k_3d, (TILE_K, head_dim))

        # Transpose K for matmul: [head_dim, TILE_K]
        k_t = ct.transpose(k_tile)

        # Convert to TF32 for tensor core
        q_tf32 = ct.astype(q, ct.tfloat32)
        k_tf32 = ct.astype(k_t, ct.tfloat32)

        # [1, head_dim] @ [head_dim, TILE_K] -> [1, TILE_K]
        scores = ct.zeros((1, TILE_K), dtype=ct.float32)
        scores = ct.mma(q_tf32, k_tf32, scores)
        # Use ct.mul to maintain float32 dtype
        scores = ct.mul(scores, qk_scale)

        # Online softmax update
        m_curr = ct.max(scores, axis=-1, keepdims=True)  # [1, 1]
        # Use ct.maximum instead of Python max to keep dtype as float32
        m_new = ct.maximum(m_prev, m_curr)

        # Rescale previous accumulator - use astype to ensure float32
        scale_prev = ct.exp(ct.sub(m_prev, m_new))
        scale_prev_fp32 = ct.astype(scale_prev, ct.float32)
        acc = ct.mul(acc, scale_prev_fp32)
        l_prev = ct.mul(l_prev, scale_prev_fp32)

        # Compute attention weights for this tile
        p = ct.exp(ct.sub(scores, m_new))  # [1, TILE_K]
        l_curr = ct.sum(p, axis=-1, keepdims=True)  # [1, 1]
        l_curr = ct.astype(l_curr, ct.float32)
        l_prev = ct.add(l_prev, l_curr)

        # Load V tile [1, TILE_K, head_dim] - use same k_start offset
        v_3d = ct.load(V, index=(kv_head, k_start, 0), shape=(1, TILE_K, head_dim), latency=3)
        v_tile = ct.reshape(v_3d, (TILE_K, head_dim))

        # Accumulate: acc += p @ V
        # [1, TILE_K] @ [TILE_K, head_dim] -> [1, head_dim]
        p_tf32 = ct.astype(p, ct.tfloat32)
        v_tf32 = ct.astype(v_tile, ct.tfloat32)
        # Ensure acc is float32 before MMA
        acc = ct.astype(acc, ct.float32)
        acc = ct.mma(p_tf32, v_tf32, acc)

        m_prev = ct.astype(m_new, ct.float32)

    # Final normalization
    acc = ct.truediv(acc, l_prev)

    # Store output - reshape to 3D
    acc_3d = ct.reshape(acc, (1, 1, head_dim))
    ct.store(Out, index=(bid, 0, 0), tile=acc_3d)


def decode_attention(
    q: cp.ndarray,
    k: cp.ndarray,
    v: cp.ndarray,
    scale: Optional[float] = None,
) -> cp.ndarray:
    """
    Optimized attention for decode mode (seq_q=1).

    For small seq_k (<= 256), uses simple cuBLAS matmul which is faster.
    For larger seq_k, uses tiled CuTile kernel with online softmax.

    Args:
        q: Query [batch, num_heads, 1, head_dim]
        k: Key [batch, num_kv_heads, seq_k, head_dim]
        v: Value [batch, num_kv_heads, seq_k, head_dim]
        scale: Optional scale factor (default: 1/sqrt(head_dim))

    Returns:
        Output [batch, num_heads, 1, head_dim]
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, num_kv_heads, seq_k, _ = k.shape

    assert seq_q == 1, "decode_attention only supports seq_q=1"

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    query_group_size = num_heads // num_kv_heads

    # For very small seq_k, cuBLAS is faster than tiled kernel
    # Based on benchmarks, tiled kernel is faster for seq_k >= 32
    if seq_k <= 32:
        return _decode_attention_cublas(q, k, v, scale, query_group_size)

    # For larger seq_k, use tiled CuTile kernel
    return _decode_attention_tiled(q, k, v, scale, query_group_size)


def _decode_attention_cublas(
    q: cp.ndarray,
    k: cp.ndarray,
    v: cp.ndarray,
    scale: float,
    query_group_size: int,
) -> cp.ndarray:
    """Fast decode attention using cuBLAS for small seq_k."""
    batch, num_heads, _, head_dim = q.shape
    _, num_kv_heads, seq_k, _ = k.shape

    # Expand KV for GQA
    if query_group_size > 1:
        k = cp.repeat(k, query_group_size, axis=1)
        v = cp.repeat(v, query_group_size, axis=1)

    # Q @ K^T
    scores = cp.matmul(q, cp.swapaxes(k, -2, -1)) * scale

    # Softmax
    scores_max = cp.max(scores, axis=-1, keepdims=True)
    exp_scores = cp.exp(scores - scores_max)
    attn = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)

    # @ V
    return cp.matmul(attn, v)


def _decode_attention_tiled(
    q: cp.ndarray,
    k: cp.ndarray,
    v: cp.ndarray,
    scale: float,
    query_group_size: int,
) -> cp.ndarray:
    """Tiled CuTile decode attention for large seq_k."""
    batch, num_heads, _, head_dim = q.shape
    _, num_kv_heads, seq_k, _ = k.shape

    # Flatten batch and heads, keep seq_q=1 dimension
    q_flat = q.reshape(batch * num_heads, 1, head_dim).astype(cp.float32)
    k_flat = k.reshape(batch * num_kv_heads, seq_k, head_dim).astype(cp.float32)
    v_flat = v.reshape(batch * num_kv_heads, seq_k, head_dim).astype(cp.float32)

    # Ensure contiguous
    q_flat = cp.ascontiguousarray(q_flat)
    k_flat = cp.ascontiguousarray(k_flat)
    v_flat = cp.ascontiguousarray(v_flat)

    # Output
    out_flat = cp.zeros((batch * num_heads, 1, head_dim), dtype=cp.float32)

    # Tile size for K dimension
    TILE_K = 32

    # Pad seq_k to multiple of TILE_K
    seq_k_padded = ((seq_k + TILE_K - 1) // TILE_K) * TILE_K

    if seq_k != seq_k_padded:
        k_padded = cp.zeros((batch * num_kv_heads, seq_k_padded, head_dim), dtype=cp.float32)
        v_padded = cp.zeros((batch * num_kv_heads, seq_k_padded, head_dim), dtype=cp.float32)
        k_padded[:, :seq_k, :] = k_flat
        v_padded[:, :seq_k, :] = v_flat
    else:
        k_padded = k_flat
        v_padded = v_flat

    # Launch kernel
    grid = (batch * num_heads,)

    ct.launch(
        get_stream(),
        grid,
        decode_attention_kernel,
        (q_flat, k_padded, v_padded, out_flat, scale, head_dim, seq_k_padded, TILE_K, query_group_size)
    )

    # Reshape output
    return out_flat.reshape(batch, num_heads, 1, head_dim)


# Simpler version using existing matmul for comparison
def decode_attention_simple(
    q: cp.ndarray,
    k: cp.ndarray,
    v: cp.ndarray,
    scale: Optional[float] = None,
) -> cp.ndarray:
    """
    Simple decode attention using CuPy matmul.
    For comparison baseline.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, num_kv_heads, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Expand KV for GQA
    repeats = num_heads // num_kv_heads
    k = cp.repeat(k, repeats, axis=1)
    v = cp.repeat(v, repeats, axis=1)

    # Q @ K^T
    scores = cp.matmul(q, cp.swapaxes(k, -2, -1)) * scale

    # Softmax
    scores_max = cp.max(scores, axis=-1, keepdims=True)
    exp_scores = cp.exp(scores - scores_max)
    attn = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)

    # @ V
    return cp.matmul(attn, v)


if __name__ == "__main__":
    import time

    print("Testing decode attention...")

    batch = 1
    num_heads = 16
    num_kv_heads = 4
    seq_k = 100
    head_dim = 128

    q = cp.random.randn(batch, num_heads, 1, head_dim).astype(cp.float32)
    k = cp.random.randn(batch, num_kv_heads, seq_k, head_dim).astype(cp.float32)
    v = cp.random.randn(batch, num_kv_heads, seq_k, head_dim).astype(cp.float32)

    print(f"Q: {q.shape}, K: {k.shape}, V: {v.shape}")

    # Test simple version first
    out_simple = decode_attention_simple(q, k, v)
    print(f"Simple output shape: {out_simple.shape}")

    # Test CuTile version
    try:
        out_cutile = decode_attention(q, k, v)
        print(f"CuTile output shape: {out_cutile.shape}")

        # Compare
        diff = float(cp.max(cp.abs(out_simple - out_cutile)))
        print(f"Max diff: {diff:.6f}")
    except Exception as e:
        print(f"CuTile version failed: {e}")

    # Benchmark
    print("\nBenchmarking...")

    # Simple
    for _ in range(10):
        _ = decode_attention_simple(q, k, v)
    cp.cuda.Device().synchronize()

    times = []
    for _ in range(100):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        _ = decode_attention_simple(q, k, v)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    print(f"Simple: {sum(times[10:])/len(times[10:]):.4f}ms")

    # CuTile
    try:
        for _ in range(10):
            _ = decode_attention(q, k, v)
        cp.cuda.Device().synchronize()

        times = []
        for _ in range(100):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            _ = decode_attention(q, k, v)
            cp.cuda.Device().synchronize()
            t1 = time.perf_counter()
            times.append((t1-t0)*1000)
        print(f"CuTile: {sum(times[10:])/len(times[10:]):.4f}ms")
    except Exception as e:
        print(f"CuTile benchmark failed: {e}")
