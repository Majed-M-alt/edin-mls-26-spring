"""
Benchmark script to compare CuTile decode vs PyTorch performance.
"""
import time
import cupy as cp
import numpy as np

# Configuration for decoder (matching GLM-ASR-Nano)
BATCH = 1
NUM_HEADS = 28
NUM_KV_HEADS = 4
SEQ_K = 100  # KV cache length
HIDDEN_SIZE = 3584
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 128
INTERMEDIATE_SIZE = 18944

def benchmark_attention():
    """Benchmark attention for decode mode (seq_q=1)."""
    print("\n=== Attention Benchmark (seq_q=1) ===")

    # Create test data
    q = cp.random.randn(BATCH, NUM_HEADS, 1, HEAD_DIM).astype(cp.float32)
    k = cp.random.randn(BATCH, NUM_KV_HEADS, SEQ_K, HEAD_DIM).astype(cp.float32)
    v = cp.random.randn(BATCH, NUM_KV_HEADS, SEQ_K, HEAD_DIM).astype(cp.float32)

    # Method 1: CuPy matmul (baseline)
    def attn_cupy(q, k, v):
        scale = 1.0 / np.sqrt(HEAD_DIM)
        # Expand KV for GQA
        repeats = NUM_HEADS // NUM_KV_HEADS
        k_exp = cp.repeat(k, repeats, axis=1)
        v_exp = cp.repeat(v, repeats, axis=1)
        # Q @ K^T
        scores = cp.matmul(q, cp.swapaxes(k_exp, -2, -1)) * scale
        # Softmax
        scores_max = cp.max(scores, axis=-1, keepdims=True)
        exp_scores = cp.exp(scores - scores_max)
        attn = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)
        return cp.matmul(attn, v_exp)

    # Warmup
    for _ in range(10):
        _ = attn_cupy(q, k, v)
    cp.cuda.Device().synchronize()

    # Benchmark CuPy
    times = []
    for _ in range(100):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        _ = attn_cupy(q, k, v)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    print(f"CuPy matmul attention: {np.mean(times[10:]):.4f}ms")

    # Method 2: FlashAttention
    try:
        from flash_attention import flash_attention

        # Warmup
        for _ in range(10):
            _ = flash_attention(q, k, v, causal=False)
        cp.cuda.Device().synchronize()

        times = []
        for _ in range(100):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            _ = flash_attention(q, k, v, causal=False)
            cp.cuda.Device().synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        print(f"FlashAttention: {np.mean(times[10:]):.4f}ms")
    except Exception as e:
        print(f"FlashAttention failed: {e}")

    # Method 3: CuTile decode attention
    try:
        from decode_attention import decode_attention

        # Warmup
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
            times.append((t1 - t0) * 1000)
        print(f"CuTile decode attention: {np.mean(times[10:]):.4f}ms")
    except Exception as e:
        print(f"CuTile decode attention failed: {e}")


def benchmark_rope():
    """Benchmark RoPE application."""
    print("\n=== RoPE Benchmark ===")

    from rope import RotaryEmbedding, apply_rotary_pos_emb, apply_rotary_pos_emb_decode

    # Create test data
    q = cp.random.randn(BATCH, NUM_HEADS, 1, HEAD_DIM).astype(cp.float32)
    k = cp.random.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM).astype(cp.float32)

    rope = RotaryEmbedding(dim=HEAD_DIM, max_position_embeddings=8192, base=500000.0)
    cos, sin = rope(q)

    # Warmup CuPy
    for _ in range(10):
        _ = apply_rotary_pos_emb(q, k, cos, sin)
    cp.cuda.Device().synchronize()

    # Benchmark CuPy
    times = []
    for _ in range(100):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        _ = apply_rotary_pos_emb(q, k, cos, sin)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    print(f"RoPE (CuPy elementwise): {np.mean(times[10:]):.4f}ms")

    # Benchmark CuTile decode
    try:
        # Warmup
        for _ in range(10):
            _ = apply_rotary_pos_emb_decode(q, k, cos, sin)
        cp.cuda.Device().synchronize()

        times = []
        for _ in range(100):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            _ = apply_rotary_pos_emb_decode(q, k, cos, sin)
            cp.cuda.Device().synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        print(f"RoPE (CuTile decode): {np.mean(times[10:]):.4f}ms")
    except Exception as e:
        print(f"CuTile RoPE decode failed: {e}")


def benchmark_mlp():
    """Benchmark MLP for decode mode (M=1)."""
    print("\n=== MLP Benchmark (M=1) ===")

    from layers import MLP

    # Create test data
    x = cp.random.randn(BATCH, 1, HIDDEN_SIZE).astype(cp.float32)

    mlp = MLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, activation="silu", use_gating=True)

    # Test cuBLAS FP16
    MLP.USE_CUBLAS_FP16 = True
    MLP.USE_FP16 = False
    MLP.FUSED = True

    # Warmup
    for _ in range(10):
        _ = mlp(x)
    cp.cuda.Device().synchronize()

    times = []
    for _ in range(100):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        _ = mlp(x)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    print(f"MLP cuBLAS FP16: {np.mean(times[10:]):.4f}ms")

    # Test CuTile FP16
    MLP.USE_CUBLAS_FP16 = False
    MLP.USE_FP16 = True

    # Warmup
    for _ in range(10):
        _ = mlp(x)
    cp.cuda.Device().synchronize()

    times = []
    for _ in range(100):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        _ = mlp(x)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    print(f"MLP CuTile FP16 fused: {np.mean(times[10:]):.4f}ms")


def benchmark_linear():
    """Benchmark Linear layer for decode mode (M=1)."""
    print("\n=== Linear Benchmark (M=1) ===")

    from layers import Linear

    # Create test data
    x = cp.random.randn(BATCH, 1, HIDDEN_SIZE).astype(cp.float32)

    linear = Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)

    # Test cuBLAS
    Linear.BACKEND = 'cublas'

    # Warmup
    for _ in range(10):
        _ = linear(x)
    cp.cuda.Device().synchronize()

    times = []
    for _ in range(100):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        _ = linear(x)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    print(f"Linear cuBLAS FP32: {np.mean(times[10:]):.4f}ms")

    # Test cuBLAS FP16
    Linear.BACKEND = 'cublas_fp16'

    # Warmup
    for _ in range(10):
        _ = linear(x)
    cp.cuda.Device().synchronize()

    times = []
    for _ in range(100):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        _ = linear(x)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    print(f"Linear cuBLAS FP16: {np.mean(times[10:]):.4f}ms")


def benchmark_decoder_layer():
    """Benchmark full decoder layer for decode mode."""
    print("\n=== Decoder Layer Benchmark (seq_q=1) ===")

    from model import DecoderLayer
    from rope import RotaryEmbedding
    from layers import Linear, MLP

    # Set optimal backends
    Linear.BACKEND = 'cublas_fp16'
    MLP.USE_CUBLAS_FP16 = True
    MLP.FUSED = True

    # Configuration
    rope = RotaryEmbedding(
        dim=HEAD_DIM,
        max_position_embeddings=8192,
        base=500000.0
    )

    layer = DecoderLayer(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        rope=rope
    )

    # Create test data
    x = cp.random.randn(BATCH, 1, HIDDEN_SIZE).astype(cp.float32)
    position_ids = cp.array([[SEQ_K]], dtype=cp.int64)

    # Create KV cache with correct shape: (batch, num_kv_heads, max_seq_len, head_dim)
    max_cache_len = SEQ_K + 100
    kv_buffer = (
        cp.zeros((BATCH, NUM_KV_HEADS, max_cache_len, HEAD_DIM), dtype=cp.float32),
        cp.zeros((BATCH, NUM_KV_HEADS, max_cache_len, HEAD_DIM), dtype=cp.float32)
    )
    # Fill the cache up to SEQ_K
    kv_buffer[0][:, :, :SEQ_K, :] = cp.random.randn(BATCH, NUM_KV_HEADS, SEQ_K, HEAD_DIM).astype(cp.float32)
    kv_buffer[1][:, :, :SEQ_K, :] = cp.random.randn(BATCH, NUM_KV_HEADS, SEQ_K, HEAD_DIM).astype(cp.float32)

    # Warmup (more iterations for kernel compilation)
    for i in range(50):
        out, _ = layer.forward_with_kv_buffer(x, kv_buffer, SEQ_K, position_ids)
    cp.cuda.Device().synchronize()

    # Benchmark (more iterations for stability)
    times = []
    for i in range(200):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        out, _ = layer.forward_with_kv_buffer(x, kv_buffer, SEQ_K, position_ids)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    # Use median and discard first 50 iterations
    stable_times = times[50:]
    print(f"Decoder layer (median): {np.median(stable_times):.4f}ms")
    print(f"Decoder layer (mean): {np.mean(stable_times):.4f}ms")
    print(f"28 layers estimate: {np.median(stable_times) * 28:.4f}ms")


if __name__ == "__main__":
    print("=" * 60)
    print("CuTile v10 Decode Performance Benchmark")
    print("=" * 60)

    benchmark_attention()
    benchmark_rope()
    benchmark_mlp()
    benchmark_linear()
    benchmark_decoder_layer()

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)
