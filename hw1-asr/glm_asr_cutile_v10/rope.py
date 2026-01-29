"""
Pure CuTile Rotary Position Embeddings (RoPE)
End-to-end implementation using only NVIDIA CuTile kernels
"""

import cuda.tile as ct
import cupy as cp
import numpy as np
from typing import Tuple, Optional


# ============================================================================
# CuTile RoPE Kernels for Decode Mode (seq_q=1)
# ============================================================================

@ct.kernel(occupancy=4)
def rope_decode_kernel(
    q,              # Query: (batch_heads, 1, head_dim)
    k,              # Key: (batch_kv_heads, 1, head_dim)
    cos,            # Cos: (1, half_dim)
    sin,            # Sin: (1, half_dim)
    q_out,          # Output query
    k_out,          # Output key
    half_dim: ct.Constant[int],
    num_q_heads: ct.Constant[int],
    num_kv_heads: ct.Constant[int],
):
    """
    Pure CuTile RoPE for decode mode (seq_q=1).
    Handles GQA: applies RoPE to all Q heads but only corresponding KV heads.
    """
    pid = ct.bid(0)  # batch * num_q_heads

    # Which KV head corresponds to this Q head
    kv_head = pid // (num_q_heads // num_kv_heads)

    # Load Q first and second halves
    q1 = ct.load(q, index=(pid, 0, 0), shape=(1, 1, half_dim))
    q1 = ct.reshape(q1, (half_dim,))

    q2 = ct.load(q, index=(pid, 0, half_dim), shape=(1, 1, half_dim))
    q2 = ct.reshape(q2, (half_dim,))

    # Load cos/sin (same for all heads at this position)
    cos_tile = ct.load(cos, index=(0, 0), shape=(1, half_dim))
    cos_tile = ct.reshape(cos_tile, (half_dim,))

    sin_tile = ct.load(sin, index=(0, 0), shape=(1, half_dim))
    sin_tile = ct.reshape(sin_tile, (half_dim,))

    # Apply rotation to Q: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    q_rot1 = q1 * cos_tile - q2 * sin_tile
    q_rot2 = q2 * cos_tile + q1 * sin_tile

    # Store Q result
    q_rot1 = ct.reshape(q_rot1, (1, 1, half_dim))
    q_rot2 = ct.reshape(q_rot2, (1, 1, half_dim))
    ct.store(q_out, index=(pid, 0, 0), tile=q_rot1)
    ct.store(q_out, index=(pid, 0, half_dim), tile=q_rot2)


@ct.kernel(occupancy=4)
def rope_decode_k_kernel(
    k,              # Key: (batch_kv_heads, 1, head_dim)
    cos,            # Cos: (1, half_dim)
    sin,            # Sin: (1, half_dim)
    k_out,          # Output key
    half_dim: ct.Constant[int],
):
    """
    Pure CuTile RoPE for K only (decode mode).
    """
    pid = ct.bid(0)  # batch * num_kv_heads

    # Load K first and second halves
    k1 = ct.load(k, index=(pid, 0, 0), shape=(1, 1, half_dim))
    k1 = ct.reshape(k1, (half_dim,))

    k2 = ct.load(k, index=(pid, 0, half_dim), shape=(1, 1, half_dim))
    k2 = ct.reshape(k2, (half_dim,))

    # Load cos/sin
    cos_tile = ct.load(cos, index=(0, 0), shape=(1, half_dim))
    cos_tile = ct.reshape(cos_tile, (half_dim,))

    sin_tile = ct.load(sin, index=(0, 0), shape=(1, half_dim))
    sin_tile = ct.reshape(sin_tile, (half_dim,))

    # Apply rotation to K
    k_rot1 = k1 * cos_tile - k2 * sin_tile
    k_rot2 = k2 * cos_tile + k1 * sin_tile

    # Store K result
    k_rot1 = ct.reshape(k_rot1, (1, 1, half_dim))
    k_rot2 = ct.reshape(k_rot2, (1, 1, half_dim))
    ct.store(k_out, index=(pid, 0, 0), tile=k_rot1)
    ct.store(k_out, index=(pid, 0, half_dim), tile=k_rot2)


# Legacy CuPy kernel for comparison
_rope_kernel = cp.ElementwiseKernel(
    'float32 x1, float32 x2, float32 cos_val, float32 sin_val',
    'float32 y1, float32 y2',
    '''
    y1 = x1 * cos_val - x2 * sin_val;
    y2 = x2 * cos_val + x1 * sin_val;
    ''',
    'rope_fused_optimized'
)


def get_stream():
    """Get current CUDA stream pointer."""
    return cp.cuda.get_current_stream().ptr


# ============================================================================
# CuTile Kernels for RoPE
# ============================================================================

@ct.kernel
def compute_freqs_kernel(
    positions,      # Input: (seq_len,)
    inv_freq,       # Input: (rotary_dim // 2,)
    cos_out,        # Output: (seq_len, rotary_dim)
    sin_out,        # Output: (seq_len, rotary_dim)
    seq_len: ct.Constant[int],
    half_dim: ct.Constant[int]
):
    """Compute cos and sin for rotary embeddings."""
    pid = ct.bid(0)  # position index

    # Load position
    pos = ct.load(positions, index=(pid,), shape=())

    # Load all inverse frequencies
    inv_freq_tile = ct.load(inv_freq, index=(0,), shape=(half_dim,))

    # Compute freqs = position * inv_freq
    freqs = pos * inv_freq_tile

    # Compute cos and sin, repeat for full dimension
    cos_half = ct.cos(freqs)
    sin_half = ct.sin(freqs)

    # Concatenate: [cos_half, cos_half] and [sin_half, sin_half]
    # ct.cat takes (tuple_of_tiles, axis)
    cos_full = ct.cat((cos_half, cos_half), 0)
    sin_full = ct.cat((sin_half, sin_half), 0)

    # Store
    cos_full = ct.reshape(cos_full, (1, half_dim * 2))
    sin_full = ct.reshape(sin_full, (1, half_dim * 2))

    ct.store(cos_out, index=(pid, 0), tile=cos_full)
    ct.store(sin_out, index=(pid, 0), tile=sin_full)


@ct.kernel
def apply_rope_kernel(
    q,              # Query: (batch_heads, seq_len, head_dim)
    k,              # Key: (batch_heads, seq_len, head_dim)
    cos,            # Cos: (seq_len, half_dim)
    sin,            # Sin: (seq_len, half_dim)
    q_out,          # Output query
    k_out,          # Output key
    head_dim: ct.Constant[int],
    half_dim: ct.Constant[int]
):
    """
    Apply rotary position embeddings to Q and K.
    Assumes full rotary (rotary_dim == head_dim).
    half_dim = head_dim // 2
    """
    pid_bh = ct.bid(0)  # batch * heads
    pid_s = ct.bid(1)   # sequence position

    # Load Q first half and second half separately
    q1 = ct.load(q, index=(pid_bh, pid_s, 0), shape=(1, 1, half_dim))
    q1 = ct.reshape(q1, (half_dim,))

    q2 = ct.load(q, index=(pid_bh, pid_s, half_dim), shape=(1, 1, half_dim))
    q2 = ct.reshape(q2, (half_dim,))

    # Load K first half and second half
    k1 = ct.load(k, index=(pid_bh, pid_s, 0), shape=(1, 1, half_dim))
    k1 = ct.reshape(k1, (half_dim,))

    k2 = ct.load(k, index=(pid_bh, pid_s, half_dim), shape=(1, 1, half_dim))
    k2 = ct.reshape(k2, (half_dim,))

    # Load cos/sin for this position
    cos_tile = ct.load(cos, index=(pid_s, 0), shape=(1, half_dim))
    cos_tile = ct.reshape(cos_tile, (half_dim,))

    sin_tile = ct.load(sin, index=(pid_s, 0), shape=(1, half_dim))
    sin_tile = ct.reshape(sin_tile, (half_dim,))

    # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    q_rot1 = q1 * cos_tile - q2 * sin_tile
    q_rot2 = q2 * cos_tile + q1 * sin_tile

    k_rot1 = k1 * cos_tile - k2 * sin_tile
    k_rot2 = k2 * cos_tile + k1 * sin_tile

    # Store results in two parts
    q_rot1 = ct.reshape(q_rot1, (1, 1, half_dim))
    q_rot2 = ct.reshape(q_rot2, (1, 1, half_dim))
    k_rot1 = ct.reshape(k_rot1, (1, 1, half_dim))
    k_rot2 = ct.reshape(k_rot2, (1, 1, half_dim))

    ct.store(q_out, index=(pid_bh, pid_s, 0), tile=q_rot1)
    ct.store(q_out, index=(pid_bh, pid_s, half_dim), tile=q_rot2)
    ct.store(k_out, index=(pid_bh, pid_s, 0), tile=k_rot1)
    ct.store(k_out, index=(pid_bh, pid_s, half_dim), tile=k_rot2)


# ============================================================================
# RoPE Classes
# ============================================================================

class RotaryEmbedding:
    """Rotary Position Embedding using pure CuTile."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        # Calculate rotary dimension
        self.rotary_dim = int(dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)  # Must be even

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (cp.arange(0, self.rotary_dim, 2, dtype=cp.float32) / self.rotary_dim)
        )
        self.inv_freq = inv_freq

        # Pre-compute cos/sin cache
        self._update_cache(max_position_embeddings)

        # Decode mode cache for single positions (avoids Python indexing overhead)
        self._decode_pos_cache = {}
        self._decode_cache_max_size = 1024  # Keep last N positions cached

    def _update_cache(self, seq_len: int):
        """Pre-compute cos and sin using CuTile kernel."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2

        positions = cp.arange(seq_len, dtype=cp.float32)
        cos_cache = cp.empty((seq_len, self.rotary_dim), dtype=cp.float32)
        sin_cache = cp.empty((seq_len, self.rotary_dim), dtype=cp.float32)

        ct.launch(
            get_stream(),
            (seq_len,),
            compute_freqs_kernel,
            (positions, self.inv_freq, cos_cache, sin_cache, seq_len, half_dim)
        )

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def get_decode_cos_sin(self, position: int) -> Tuple[cp.ndarray, cp.ndarray]:
        """Get pre-allocated cos/sin for a single decode position (fast path)."""
        if position not in self._decode_pos_cache:
            if position >= self.max_seq_len_cached:
                self._update_cache(position + 1024)

            # Cache contiguous (1, half_dim) slices for this position
            half_dim = self.rotary_dim // 2
            cos = cp.ascontiguousarray(self.cos_cached[position:position+1, :half_dim])
            sin = cp.ascontiguousarray(self.sin_cached[position:position+1, :half_dim])

            # Limit cache size
            if len(self._decode_pos_cache) >= self._decode_cache_max_size:
                # Remove oldest entries
                keys_to_remove = sorted(self._decode_pos_cache.keys())[:100]
                for k in keys_to_remove:
                    del self._decode_pos_cache[k]

            self._decode_pos_cache[position] = (cos, sin)

        return self._decode_pos_cache[position]

    def __call__(
        self,
        x: cp.ndarray,
        position_ids: Optional[cp.ndarray] = None
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Get cos and sin for given positions."""
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len)

        if position_ids is not None:
            # Fast path for decode mode (seq_len=1, scalar position)
            if seq_len == 1 and position_ids.size == 1:
                pos = int(position_ids.flat[0])
                cos, sin = self.get_decode_cos_sin(pos)
                return cos.astype(x.dtype), sin.astype(x.dtype)

            # General path: position_ids: (batch, seq_len)
            cos = self.cos_cached[position_ids].astype(x.dtype)
            sin = self.sin_cached[position_ids].astype(x.dtype)
            # Squeeze batch dim if single batch for simpler broadcasting
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]  # (seq_len, rotary_dim)
                sin = sin[0]
        else:
            cos = self.cos_cached[:seq_len].astype(x.dtype)
            sin = self.sin_cached[:seq_len].astype(x.dtype)

        return cos, sin


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


# Maximum dimension for CuTile RoPE kernel
MAX_ROPE_DIM = 256


def _apply_rope_single(
    x: cp.ndarray,
    cos: cp.ndarray,
    sin: cp.ndarray,
    half_dim: int,
    head_dim: int
) -> cp.ndarray:
    """Apply RoPE to a single tensor (Q or K) using optimized fused kernel."""
    batch, num_heads, seq_len, _ = x.shape

    # cos/sin should already be (seq_len, half_dim) from caller
    # Just truncate seq_len if needed
    cos = cos[:seq_len]
    sin = sin[:seq_len]

    # Broadcast cos/sin for the fused kernel
    cos_expanded = cp.broadcast_to(cos[None, None, :, :], (batch, num_heads, seq_len, half_dim))
    sin_expanded = cp.broadcast_to(sin[None, None, :, :], (batch, num_heads, seq_len, half_dim))

    # Split input
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:half_dim * 2]

    # Allocate output for rotated parts
    x1_rot = cp.empty_like(x1)
    x2_rot = cp.empty_like(x2)

    # Apply fused RoPE kernel (single kernel launch)
    _rope_kernel(x1, x2, cos_expanded, sin_expanded, x1_rot, x2_rot)

    if head_dim > half_dim * 2:
        x_pass = x[..., half_dim * 2:]
        return cp.concatenate([x1_rot, x2_rot, x_pass], axis=-1)
    else:
        return cp.concatenate([x1_rot, x2_rot], axis=-1)


def apply_rotary_pos_emb(
    q: cp.ndarray,
    k: cp.ndarray,
    cos: cp.ndarray,
    sin: cp.ndarray,
    rotary_dim: Optional[int] = None
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Apply rotary position embeddings.

    Args:
        q: Query (batch, num_q_heads, seq_len, head_dim)
        k: Key (batch, num_kv_heads, seq_len, head_dim) - may have different num heads
        cos: Cosine values (seq_len, rotary_dim)
        sin: Sine values (seq_len, rotary_dim)
        rotary_dim: Dimension to apply rotation (default: full head_dim)
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape

    if rotary_dim is None:
        rotary_dim = head_dim

    half_dim = rotary_dim // 2

    # Ensure cos/sin are correct shape
    if cos.shape[1] > half_dim:
        cos = cos[:, :half_dim]
        sin = sin[:, :half_dim]

    cos = cp.ascontiguousarray(cos.astype(cp.float32))
    sin = cp.ascontiguousarray(sin.astype(cp.float32))

    # Apply RoPE to Q and K separately (handles different head counts)
    q_out = _apply_rope_single(q, cos, sin, half_dim, head_dim)
    k_out = _apply_rope_single(k, cos, sin, half_dim, head_dim)

    return q_out.astype(q.dtype), k_out.astype(k.dtype)


def apply_partial_rotary_pos_emb(
    q: cp.ndarray,
    k: cp.ndarray,
    cos: cp.ndarray,
    sin: cp.ndarray,
    rotary_dim: int
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Apply rotary embeddings to partial dimensions."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)


def apply_rotary_pos_emb_decode(
    q: cp.ndarray,
    k: cp.ndarray,
    cos: cp.ndarray,
    sin: cp.ndarray,
    rotary_dim: Optional[int] = None
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Apply RoPE for decode mode (seq_len=1) using fused CuPy kernel.

    Args:
        q: Query (batch, num_q_heads, 1, head_dim)
        k: Key (batch, num_kv_heads, 1, head_dim)
        cos: Cosine values (1, half_dim) - already pre-sliced from RotaryEmbedding
        sin: Sine values (1, half_dim) - already pre-sliced from RotaryEmbedding
        rotary_dim: Dimension to apply rotation (default: full head_dim)

    Returns:
        (q_rotated, k_rotated) tuple
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape

    assert seq_len == 1, "apply_rotary_pos_emb_decode only supports seq_len=1"

    if rotary_dim is None:
        rotary_dim = head_dim

    half_dim = rotary_dim // 2
    q_dtype = q.dtype

    # Ensure cos/sin have correct shape (1, half_dim)
    if cos.shape[-1] != half_dim:
        cos = cos[..., :half_dim]
        sin = sin[..., :half_dim]

    # Ensure float32 for computation
    cos = cp.ascontiguousarray(cos.astype(cp.float32))
    sin = cp.ascontiguousarray(sin.astype(cp.float32))

    # Process Q: broadcast cos/sin to (batch, num_q_heads, 1, half_dim)
    cos_q = cp.broadcast_to(cos[None, None, :, :], (batch, num_q_heads, 1, half_dim))
    sin_q = cp.broadcast_to(sin[None, None, :, :], (batch, num_q_heads, 1, half_dim))

    q1 = q[..., :half_dim].astype(cp.float32)
    q2 = q[..., half_dim:half_dim * 2].astype(cp.float32)

    q1_rot = cp.empty_like(q1)
    q2_rot = cp.empty_like(q2)
    _rope_kernel(q1, q2, cos_q, sin_q, q1_rot, q2_rot)

    if head_dim > half_dim * 2:
        q_pass = q[..., half_dim * 2:]
        q_out = cp.concatenate([q1_rot, q2_rot, q_pass], axis=-1)
    else:
        q_out = cp.concatenate([q1_rot, q2_rot], axis=-1)

    # Process K: broadcast cos/sin to (batch, num_kv_heads, 1, half_dim)
    cos_k = cp.broadcast_to(cos[None, None, :, :], (batch, num_kv_heads, 1, half_dim))
    sin_k = cp.broadcast_to(sin[None, None, :, :], (batch, num_kv_heads, 1, half_dim))

    k1 = k[..., :half_dim].astype(cp.float32)
    k2 = k[..., half_dim:half_dim * 2].astype(cp.float32)

    k1_rot = cp.empty_like(k1)
    k2_rot = cp.empty_like(k2)
    _rope_kernel(k1, k2, cos_k, sin_k, k1_rot, k2_rot)

    if head_dim > half_dim * 2:
        k_pass = k[..., half_dim * 2:]
        k_out = cp.concatenate([k1_rot, k2_rot, k_pass], axis=-1)
    else:
        k_out = cp.concatenate([k1_rot, k2_rot], axis=-1)

    return q_out.astype(q_dtype), k_out.astype(q_dtype)


if __name__ == "__main__":
    print("Testing Pure CuTile RoPE...")

    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    # Create RoPE
    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=1024)

    # Create Q, K
    q = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)
    k = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)

    # Get cos/sin
    cos, sin = rope(q)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")

    # Apply rotation
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"Q rotated shape: {q_rot.shape}")
    print(f"K rotated shape: {k_rot.shape}")

    # Test partial rotary (50%)
    print("\nTesting partial RoPE (50%):")
    rope_partial = RotaryEmbedding(dim=head_dim, partial_rotary_factor=0.5)
    cos_p, sin_p = rope_partial(q)
    q_rot_p, k_rot_p = apply_partial_rotary_pos_emb(q, k, cos_p, sin_p, head_dim // 2)
    print(f"Q rotated (partial) shape: {q_rot_p.shape}")

    print("\nPure CuTile RoPE working!")
