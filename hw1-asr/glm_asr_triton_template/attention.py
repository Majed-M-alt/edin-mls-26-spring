"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_q, seq_k, head_dim, scale,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """True FlashAttention: Iterating over K to keep SRAM footprint small."""
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    # 1. Load Q block
    q_ptrs = q_ptr + pid_b * stride_qb + offs_q[:, None] * stride_qq + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_q[:, None] < seq_q) & (offs_d[None, :] < head_dim), other=0.0)

    # 2. Initialize running softmax stats
    m_i = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    # 3. Blockwise Loop over K and V
    for k_idx in range(0, seq_k, BLOCK_K):
        k_offs = k_idx + offs_k

        # Load K transposed
        k_ptrs = k_ptr + pid_b * stride_kb + k_offs[None, :] * stride_kk + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=(k_offs[None, :] < seq_k) & (offs_d[:, None] < head_dim), other=0.0)

        # Load V
        v_ptrs = v_ptr + pid_b * stride_vb + k_offs[:, None] * stride_vk + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(k_offs[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0)

        # Q @ K^T
        qk = tl.dot(q, k) * scale
        qk = tl.where(k_offs[None, :] < seq_k, qk, float("-inf"))

        if IS_CAUSAL:
            qk = tl.where(k_offs[None, :] <= offs_q[:, None], qk, float("-inf"))

        # Streaming Softmax Math
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)

        # Update running denominator and max
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

        # Multiply by V and accumulate
        acc = acc * alpha[:, None] + tl.dot(p, v)

    # 4. Normalize and Store
    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + pid_b * stride_ob + offs_q[:, None] * stride_oq + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=(offs_q[:, None] < seq_q) & (offs_d[None, :] < head_dim))

# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    # Fallback to PyTorch only if not CUDA or if a complex 4D attention_mask is passed
    if not q.is_cuda or attention_mask is not None:
        scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
        if is_causal:
            mask = torch.triu(torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device), diagonal=1) * -1e9
            scores = scores + mask[None, None, :, :]
        if attention_mask is not None:
            scores = scores + attention_mask

        scores = scores - torch.max(scores, dim=-1, keepdim=True).values
        attn_weights = torch.exp(scores)
        attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
        return torch.einsum("bnqk,bnkd->bnqd", attn_weights, v).to(q.dtype)

    # Ensure tensors are contiguous for Triton
    q_flat = q.reshape(batch * num_heads, seq_q, head_dim).contiguous().to(torch.float32)
    k_flat = k.reshape(batch * num_heads, seq_k, head_dim).contiguous().to(torch.float32)
    v_flat = v.reshape(batch * num_heads, seq_k, head_dim).contiguous().to(torch.float32)

    output = torch.empty_like(q_flat)

    # 1. Shrink block sizes to fit within SRAM limits
    BLOCK_Q = 32
    BLOCK_K = 32
    head_dim_padded = next_power_of_two(head_dim)

    grid = (batch * num_heads, triton.cdiv(seq_q, BLOCK_Q))

    flash_attention_kernel[grid](
        q_flat, k_flat, v_flat, output,
        seq_q, seq_k, head_dim, float(scale),
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        IS_CAUSAL=is_causal,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        BLOCK_D=head_dim_padded,
        num_warps=4,   # 2. Safely limit thread warps
        num_stages=2   # 3. Limit pipeline memory footprint
    )

    return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)    


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2 :] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    print("\nOutput statistics:")
    print(f"  Mean: {float(output.mean()):.4f}")
    print(f"  Std:  {float(output.std()):.4f}")
    print(f"  Min:  {float(output.min()):.4f}")
    print(f"  Max:  {float(output.max()):.4f}")

    print("\nTriton Attention working!")
