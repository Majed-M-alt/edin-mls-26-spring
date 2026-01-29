"""
Detailed profiler for decoder layer components.
"""
import time
import cupy as cp
import numpy as np

# Configuration
BATCH = 1
NUM_HEADS = 28
NUM_KV_HEADS = 4
SEQ_K = 100
HIDDEN_SIZE = 3584
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
INTERMEDIATE_SIZE = 18944

def profile_decoder_layer():
    """Profile each component of the decoder layer."""
    from model import DecoderLayer
    from rope import RotaryEmbedding, apply_rotary_pos_emb_decode
    from layers import Linear, MLP, RMSNorm

    # Set optimal backends
    Linear.BACKEND = 'cublas_fp16'
    MLP.USE_CUBLAS_FP16 = True
    MLP.FUSED = True

    # Create components
    rope = RotaryEmbedding(dim=HEAD_DIM, max_position_embeddings=8192, base=500000.0)
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

    # KV cache
    max_cache_len = SEQ_K + 100
    kv_buffer = (
        cp.random.randn(BATCH, NUM_KV_HEADS, max_cache_len, HEAD_DIM).astype(cp.float32),
        cp.random.randn(BATCH, NUM_KV_HEADS, max_cache_len, HEAD_DIM).astype(cp.float32)
    )

    # Warmup
    for _ in range(50):
        _ = layer.forward_with_kv_buffer(x, kv_buffer, SEQ_K, position_ids)
    cp.cuda.Device().synchronize()

    # Profile individual components
    def time_op(name, op, n_iter=100):
        times = []
        for _ in range(n_iter):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            result = op()
            cp.cuda.Device().synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        avg = np.mean(times[10:])
        print(f"{name}: {avg:.4f}ms")
        return avg

    print("\n=== Component-wise Profiling ===\n")

    total = 0.0

    # 1. Input LayerNorm
    hidden_states = x
    t = time_op("input_layernorm", lambda: layer.input_layernorm(hidden_states))
    total += t
    hidden_states_normed = layer.input_layernorm(hidden_states)

    # 2. Q projection
    t = time_op("q_proj", lambda: layer.q_proj(hidden_states_normed))
    total += t
    q = layer.q_proj(hidden_states_normed)

    # 3. K projection
    t = time_op("k_proj", lambda: layer.k_proj(hidden_states_normed))
    total += t
    k = layer.k_proj(hidden_states_normed)

    # 4. V projection
    t = time_op("v_proj", lambda: layer.v_proj(hidden_states_normed))
    total += t
    v = layer.v_proj(hidden_states_normed)

    # Reshape for attention
    batch, seq_len = 1, 1
    q = q.reshape(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    # 5. Make V contiguous
    t = time_op("v_contiguous", lambda: cp.ascontiguousarray(v))
    total += t
    v = cp.ascontiguousarray(v)

    # 6. RoPE - get cos/sin
    t = time_op("rope_get_cos_sin", lambda: rope(q, position_ids))
    total += t
    cos, sin = rope(q, position_ids)

    # 7. RoPE - apply
    t = time_op("rope_apply", lambda: apply_rotary_pos_emb_decode(q, k, cos, sin))
    total += t
    q_rot, k_rot = apply_rotary_pos_emb_decode(q, k, cos, sin)

    # 8. KV cache update
    key_buffer, value_buffer = kv_buffer
    def kv_update():
        key_buffer[:, :, SEQ_K:SEQ_K+1, :] = k_rot
        value_buffer[:, :, SEQ_K:SEQ_K+1, :] = v
    t = time_op("kv_cache_update", kv_update)
    total += t
    kv_update()

    # Get K/V for attention
    k_for_attn = key_buffer[:, :, :SEQ_K+1, :]
    v_for_attn = value_buffer[:, :, :SEQ_K+1, :]

    # 9. Attention
    from attention import scaled_dot_product_attention
    t = time_op("attention", lambda: scaled_dot_product_attention(q_rot, k_for_attn, v_for_attn, is_causal=False))
    total += t
    attn_output = scaled_dot_product_attention(q_rot, k_for_attn, v_for_attn, is_causal=False)

    # 10. Transpose and reshape attention output
    def attn_reshape():
        return attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
    t = time_op("attn_reshape", attn_reshape)
    total += t
    attn_output_flat = attn_reshape()

    # 11. O projection
    t = time_op("o_proj", lambda: layer.o_proj(attn_output_flat))
    total += t
    o = layer.o_proj(attn_output_flat)

    # 12. Residual add (attention)
    def residual_add_attn():
        return hidden_states + o
    t = time_op("residual_add_attn", residual_add_attn)
    total += t
    hidden_states = residual_add_attn()

    # 13. Post-attention LayerNorm
    t = time_op("post_attn_layernorm", lambda: layer.post_attention_layernorm(hidden_states))
    total += t
    hidden_states_normed2 = layer.post_attention_layernorm(hidden_states)

    # 14. MLP
    t = time_op("mlp", lambda: layer.mlp(hidden_states_normed2))
    total += t
    mlp_out = layer.mlp(hidden_states_normed2)

    # 15. Residual add (MLP)
    def residual_add_mlp():
        return hidden_states + mlp_out
    t = time_op("residual_add_mlp", residual_add_mlp)
    total += t

    print(f"\n--- Sum of components: {total:.4f}ms ---")

    # Also time the full layer for comparison
    print("\n=== Full Layer ===")
    times = []
    for _ in range(100):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        _ = layer.forward_with_kv_buffer(x, kv_buffer, SEQ_K, position_ids)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    print(f"Full layer (actual): {np.mean(times[10:]):.4f}ms")


if __name__ == "__main__":
    profile_decoder_layer()
