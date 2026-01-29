"""
Pure CuTile GLM-ASR Model Implementation
End-to-end implementation using only NVIDIA CuTile kernels
"""

import cuda.tile as ct
import cupy as cp
import numpy as np
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

# Import CuTile components
from layers import (
    RMSNorm, LayerNorm, Linear, Embedding, MLP,
    gelu, silu, softmax, get_stream
)
from rope import RotaryEmbedding, apply_rotary_pos_emb, apply_rotary_pos_emb_decode
from attention import scaled_dot_product_attention, MultiHeadAttention
from conv import Conv1d, Conv1dSubsampler


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GlmAsrConfig:
    """Configuration for GLM-ASR model."""

    # Audio encoder
    audio_hidden_size: int = 1280
    audio_num_heads: int = 20
    audio_num_layers: int = 32
    audio_intermediate_size: int = 5120
    audio_max_position_embeddings: int = 1500

    # Text decoder
    text_hidden_size: int = 3584
    text_num_heads: int = 28
    text_num_kv_heads: int = 4
    text_num_layers: int = 28
    text_intermediate_size: int = 18944
    text_vocab_size: int = 151552
    text_max_position_embeddings: int = 8192
    text_rope_base: float = 500000.0

    # Projector
    projector_hidden_size: int = 4096  # Intermediate size in projector
    projector_pool_factor: int = 4  # Concatenate 4 audio frames

    # Generation
    pad_token_id: int = 151329
    bos_token_id: int = 151331
    eos_token_id: Union[int, List[int]] = 151336  # Can be single ID or list


# ============================================================================
# Audio Encoder Components
# ============================================================================

class AudioEncoderLayer:
    """Single transformer layer for audio encoder (pre-norm with LayerNorm + RoPE)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        rotary_dim: int = 32  # Partial RoPE: only rotate first rotary_dim dimensions
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_dim = rotary_dim

        # Layer norms
        self.self_attn_layer_norm = LayerNorm(hidden_size)
        self.final_layer_norm = LayerNorm(hidden_size)

        # Attention projections
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)

        # MLP
        self.fc1 = Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=True)

    def __call__(
        self,
        hidden_states: cp.ndarray,
        attention_mask: Optional[cp.ndarray] = None,
        position_embeddings: Optional[Tuple[cp.ndarray, cp.ndarray]] = None
    ) -> cp.ndarray:
        batch, seq_len, _ = hidden_states.shape

        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE if provided (partial rotation for audio encoder)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, rotary_dim=self.rotary_dim)

        # Attention
        attn_output = scaled_dot_product_attention(q, k, v, attention_mask)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

        # Output projection + residual
        hidden_states = self.out_proj(attn_output)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class AudioEncoder:
    """Whisper-style audio encoder using pure CuTile with RoPE."""

    def __init__(self, config: GlmAsrConfig):
        self.config = config
        self.head_dim = config.audio_hidden_size // config.audio_num_heads

        # Convolutional subsampling
        self.conv1 = Conv1d(128, config.audio_hidden_size, kernel_size=3, padding=1)
        self.conv2 = Conv1d(
            config.audio_hidden_size, config.audio_hidden_size,
            kernel_size=3, stride=2, padding=1
        )

        # Rotary position embeddings (same as HF GLM-ASR)
        # Note: audio encoder uses partial_rotary_factor=0.5
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.audio_max_position_embeddings,
            base=10000.0,
            partial_rotary_factor=0.5  # Only apply RoPE to half the dimensions
        )

        # Transformer layers (with partial RoPE)
        self.layers = [
            AudioEncoderLayer(
                config.audio_hidden_size,
                config.audio_num_heads,
                config.audio_intermediate_size,
                rotary_dim=self.rotary_emb.rotary_dim  # Pass rotary_dim for partial RoPE
            )
            for _ in range(config.audio_num_layers)
        ]

        # Final layer norm
        self.layer_norm = LayerNorm(config.audio_hidden_size)

    def __call__(
        self,
        input_features: cp.ndarray,  # (batch, features, time)
        attention_mask: Optional[cp.ndarray] = None
    ) -> cp.ndarray:
        # Convolutional feature extraction
        # input_features: (batch, mel_channels, time)
        hidden_states = gelu(self.conv1(input_features))
        hidden_states = gelu(self.conv2(hidden_states))

        # (batch, hidden, time) -> (batch, time, hidden)
        hidden_states = hidden_states.transpose(0, 2, 1)

        batch, seq_len, _ = hidden_states.shape

        # Compute RoPE position embeddings
        position_ids = cp.arange(seq_len, dtype=cp.int64)[None, :]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Transformer layers with RoPE
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_embeddings)

        # Final norm
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


# ============================================================================
# Text Decoder Components
# ============================================================================

class DecoderLayer:
    """Single transformer layer for text decoder (pre-norm with RMSNorm)."""

    # Enable fused QKV projection for decode mode (reduces kernel launches)
    # Disabled due to bug - causes empty outputs in generate_v8b
    USE_FUSED_QKV = False

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rope: RotaryEmbedding
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.rope = rope

        # Layer norms
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

        # Attention projections (no bias for Llama-style)
        self.q_proj = Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # MLP (SwiGLU)
        self.mlp = MLP(
            hidden_size, intermediate_size,
            activation="silu", use_gating=True
        )

        # Attention handler
        self.attention = MultiHeadAttention(
            hidden_size, num_heads, num_kv_heads, self.head_dim
        )

        # Fused QKV weight (lazily initialized)
        self._qkv_weight_fp16 = None
        self._q_size = num_heads * self.head_dim
        self._kv_size = num_kv_heads * self.head_dim

    def _ensure_fused_qkv_weight(self):
        """Prepare fused QKV weight for optimized decode mode."""
        if self._qkv_weight_fp16 is None:
            self._qkv_weight_fp16 = cp.concatenate([
                self.q_proj.weight.astype(cp.float16),
                self.k_proj.weight.astype(cp.float16),
                self.v_proj.weight.astype(cp.float16)
            ], axis=0)

    def __call__(
        self,
        hidden_states: cp.ndarray,
        attention_mask: Optional[cp.ndarray] = None,
        position_ids: Optional[cp.ndarray] = None,
        is_causal: bool = True,
        past_key_value: Optional[Tuple[cp.ndarray, cp.ndarray]] = None,
        use_cache: bool = False
    ) -> Union[cp.ndarray, Tuple[cp.ndarray, Tuple[cp.ndarray, cp.ndarray]]]:
        """Forward pass with optional KV cache support.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Position IDs for RoPE
            is_causal: Whether to use causal attention
            past_key_value: Optional (past_key, past_value) cache from previous step
            use_cache: Whether to return updated KV cache

        Returns:
            If use_cache=False: hidden_states
            If use_cache=True: (hidden_states, (key, value)) tuple
        """
        batch, seq_len, _ = hidden_states.shape

        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        cos, sin = self.rope(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # V8: KV cache handling
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # Concatenate past K/V with current K/V
            k = cp.concatenate([past_key, k], axis=2)
            v = cp.concatenate([past_value, v], axis=2)

        # Store current K/V for cache if needed
        if use_cache:
            present_key_value = (k, v)

        # Attention with GQA support
        # When using KV cache, only query the new positions but attend to all K/V
        attn_output = self.attention(q, k, v, attention_mask, is_causal and past_key_value is None)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

        # Output projection + residual
        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            return hidden_states, present_key_value
        return hidden_states

    def forward_with_kv_buffer(
        self,
        hidden_states: cp.ndarray,
        kv_buffer: Tuple[cp.ndarray, cp.ndarray],
        cache_pos: int,
        position_ids: cp.ndarray,
        debug: bool = False,
    ) -> Tuple[cp.ndarray, int]:
        """V8.1: Forward pass with pre-allocated KV buffer (no concatenation).

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            kv_buffer: Pre-allocated (key_buffer, value_buffer) each shape
                       (batch, num_kv_heads, max_seq_len, head_dim)
            cache_pos: Current position in the cache (start of new data)
            position_ids: Position IDs for RoPE
            debug: Print debug info

        Returns:
            (hidden_states, new_cache_pos)
        """
        batch, seq_len, _ = hidden_states.shape
        key_buffer, value_buffer = kv_buffer

        if debug:
            print(f"  [Layer] input hidden: {hidden_states[0,0,:3].get()}", flush=True)

        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if debug:
            print(f"  [Layer] after layernorm: {hidden_states[0,0,:3].get()}", flush=True)

        # V10: Use fused QKV projection for decode mode (seq_len=1)
        if DecoderLayer.USE_FUSED_QKV and seq_len == 1:
            self._ensure_fused_qkv_weight()
            x_fp16 = hidden_states.reshape(-1, self.hidden_size).astype(cp.float16)
            qkv = x_fp16 @ self._qkv_weight_fp16.T
            qkv_fp32 = qkv.astype(cp.float32)
            q = qkv_fp32[:, :self._q_size]
            k = qkv_fp32[:, self._q_size:self._q_size + self._kv_size]
            v = qkv_fp32[:, self._q_size + self._kv_size:]
            # Reshape for attention
            q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        else:
            # Standard separate projections
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            # Reshape for attention
            q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # V8.2: Make V contiguous now (it won't be processed by RoPE)
        v = cp.ascontiguousarray(v)

        # Apply RoPE (creates contiguous Q/K outputs via concatenation)
        cos, sin = self.rope(q, position_ids)

        if debug:
            print(f"  [Layer] Q before RoPE: {q[0,0,0,:3].get()}", flush=True)

        # V10: Use CuTile RoPE decode for seq_len=1 (1.7x faster)
        if seq_len == 1:
            q, k = apply_rotary_pos_emb_decode(q, k, cos, sin)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if debug:
            print(f"  [Layer] Q after RoPE: {q[0,0,0,:3].get()}", flush=True)
            print(f"  [Layer] K after RoPE: {k[0,0,0,:3].get()}", flush=True)

        # V8.1: Write to pre-allocated buffer (K is contiguous from RoPE, V is contiguous)
        new_cache_pos = cache_pos + seq_len
        key_buffer[:, :, cache_pos:new_cache_pos, :] = k
        value_buffer[:, :, cache_pos:new_cache_pos, :] = v

        # Use buffer slice for attention
        k_for_attn = key_buffer[:, :, :new_cache_pos, :]
        v_for_attn = value_buffer[:, :, :new_cache_pos, :]

        if debug:
            print(f"  [Layer] k_for_attn shape: {k_for_attn.shape}, has nan: {bool(cp.any(cp.isnan(k_for_attn)).get())}", flush=True)

        # Attention with GQA support
        # For prefill (cache_pos=0), use causal attention to prevent attending to future positions
        # For decode (cache_pos>0), no causal mask needed since KV cache only has past positions
        is_prefill = (cache_pos == 0 and seq_len > 1)
        attn_output = self.attention(q, k_for_attn, v_for_attn, None, is_prefill)

        if debug:
            print(f"  [Layer] attn_output: {attn_output[0,0,0,:3].get()}", flush=True)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

        # Output projection + residual
        hidden_states = self.o_proj(attn_output)
        hidden_states = residual + hidden_states

        if debug:
            print(f"  [Layer] after attn+residual: {hidden_states[0,0,:3].get()}", flush=True)

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if debug:
            print(f"  [Layer] after MLP: {hidden_states[0,0,:3].get()}", flush=True)

        return hidden_states, new_cache_pos


class TextDecoder:
    """Llama-style text decoder using pure CuTile."""

    def __init__(self, config: GlmAsrConfig):
        self.config = config
        self.num_layers = config.text_num_layers

        # Token embeddings
        self.embed_tokens = Embedding(config.text_vocab_size, config.text_hidden_size)

        # RoPE
        self.rope = RotaryEmbedding(
            dim=config.text_hidden_size // config.text_num_heads,
            max_position_embeddings=config.text_max_position_embeddings,
            base=config.text_rope_base
        )

        # Transformer layers
        self.layers = [
            DecoderLayer(
                config.text_hidden_size,
                config.text_num_heads,
                config.text_num_kv_heads,
                config.text_intermediate_size,
                self.rope
            )
            for _ in range(config.text_num_layers)
        ]

        # Final norm
        self.norm = RMSNorm(config.text_hidden_size)

    def __call__(
        self,
        input_ids: Optional[cp.ndarray] = None,
        inputs_embeds: Optional[cp.ndarray] = None,
        attention_mask: Optional[cp.ndarray] = None,
        position_ids: Optional[cp.ndarray] = None,
        past_key_values: Optional[List[Tuple[cp.ndarray, cp.ndarray]]] = None,
        use_cache: bool = False
    ) -> Union[cp.ndarray, Tuple[cp.ndarray, List[Tuple[cp.ndarray, cp.ndarray]]]]:
        """Forward pass with optional KV cache support.

        Args:
            input_ids: Token IDs
            inputs_embeds: Pre-computed embeddings
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_values: List of (key, value) tuples for each layer
            use_cache: Whether to return KV cache

        Returns:
            If use_cache=False: hidden_states
            If use_cache=True: (hidden_states, past_key_values)
        """
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        # Generate position ids if not provided
        batch, seq_len, _ = hidden_states.shape
        if position_ids is None:
            # If we have past_key_values, offset position_ids
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = cp.arange(past_length, past_length + seq_len, dtype=cp.int64)[None, :].repeat(batch, axis=0)

        # Transformer layers with KV cache
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if use_cache:
                hidden_states, present_kv = layer(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    is_causal=True,
                    past_key_value=past_kv,
                    use_cache=True
                )
                present_key_values.append(present_kv)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    is_causal=True,
                    past_key_value=past_kv,
                    use_cache=False
                )

        # Final norm
        hidden_states = self.norm(hidden_states)

        if use_cache:
            return hidden_states, present_key_values
        return hidden_states

    def forward_with_kv_buffers(
        self,
        inputs_embeds: cp.ndarray,
        kv_buffers: List[Tuple[cp.ndarray, cp.ndarray]],
        cache_pos: int,
        debug: bool = False,
    ) -> Tuple[cp.ndarray, int]:
        """V8.1: Forward with pre-allocated KV buffers.

        Args:
            inputs_embeds: Input embeddings
            kv_buffers: List of (key_buffer, value_buffer) for each layer
            cache_pos: Current position in the cache
            debug: Print debug info for first layer

        Returns:
            (hidden_states, new_cache_pos)
        """
        hidden_states = inputs_embeds
        batch, seq_len, _ = hidden_states.shape

        # Generate position ids
        position_ids = cp.arange(cache_pos, cache_pos + seq_len, dtype=cp.int64)[None, :].repeat(batch, axis=0)

        if debug:
            print(f"[TextDecoder] cache_pos={cache_pos}, seq_len={seq_len}, position_ids={position_ids[0].get()}", flush=True)

        # Process through layers
        new_cache_pos = cache_pos
        for i, layer in enumerate(self.layers):
            layer_debug = debug and i == 0  # Only debug first layer
            hidden_states, new_cache_pos = layer.forward_with_kv_buffer(
                hidden_states,
                kv_buffers[i],
                cache_pos,
                position_ids,
                debug=layer_debug,
            )
            if debug and i == 0:
                print(f"[TextDecoder] Layer 0 output: {hidden_states[0,0,:3].get()}", flush=True)

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, new_cache_pos

    def allocate_kv_buffers(
        self,
        batch_size: int,
        max_seq_len: int,
        dtype=cp.float32,
    ) -> List[Tuple[cp.ndarray, cp.ndarray]]:
        """Allocate KV buffers for all layers.

        Returns:
            List of (key_buffer, value_buffer) for each layer.
        """
        head_dim = self.config.text_hidden_size // self.config.text_num_heads
        num_kv_heads = self.config.text_num_kv_heads

        kv_buffers = []
        for _ in range(self.num_layers):
            key_buffer = cp.zeros((batch_size, num_kv_heads, max_seq_len, head_dim), dtype=dtype)
            value_buffer = cp.zeros((batch_size, num_kv_heads, max_seq_len, head_dim), dtype=dtype)
            kv_buffers.append((key_buffer, value_buffer))

        return kv_buffers


# ============================================================================
# Multi-Modal Projector
# ============================================================================

class MultiModalProjector:
    """Projects audio features to text embedding space with frame pooling."""

    def __init__(self, config: GlmAsrConfig):
        self.pool_factor = config.projector_pool_factor
        # After pooling: audio_hidden_size * pool_factor -> projector_hidden_size
        pooled_dim = config.audio_hidden_size * self.pool_factor
        self.linear_1 = Linear(pooled_dim, config.projector_hidden_size, bias=True)
        self.act = gelu
        self.linear_2 = Linear(config.projector_hidden_size, config.text_hidden_size, bias=True)

    def _pool_frames(self, audio_features: cp.ndarray) -> cp.ndarray:
        """Pool audio frames by concatenating consecutive frames.

        Args:
            audio_features: (batch, seq_len, hidden_size) or (seq_len, hidden_size)

        Returns:
            Pooled features: (batch, seq_len // pool_factor, hidden_size * pool_factor)
        """
        if audio_features.ndim == 2:
            # (seq_len, hidden_size) -> add batch dimension
            audio_features = audio_features[None, :, :]
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch, seq_len, hidden_size = audio_features.shape

        # Truncate to multiple of pool_factor
        new_seq_len = (seq_len // self.pool_factor) * self.pool_factor
        audio_features = audio_features[:, :new_seq_len, :]

        # Reshape to concatenate frames
        # (batch, seq_len, hidden) -> (batch, seq_len // pool, pool * hidden)
        audio_features = audio_features.reshape(
            batch, new_seq_len // self.pool_factor, self.pool_factor * hidden_size
        )

        if squeeze_batch:
            audio_features = audio_features[0]

        return audio_features

    def __call__(self, audio_features: cp.ndarray) -> cp.ndarray:
        # Pool consecutive frames
        pooled = self._pool_frames(audio_features)
        # Project through MLP
        hidden_states = self.linear_1(pooled)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# ============================================================================
# Full Model
# ============================================================================

class GlmAsrModel:
    """GLM-ASR model using pure CuTile kernels."""

    def __init__(self, config: GlmAsrConfig):
        self.config = config

        # Components
        self.audio_encoder = AudioEncoder(config)
        self.multi_modal_projector = MultiModalProjector(config)
        self.text_decoder = TextDecoder(config)

        # LM head (tied with embedding)
        self.lm_head = Linear(config.text_hidden_size, config.text_vocab_size, bias=False)

    def encode_audio(
        self,
        input_features: cp.ndarray,
        input_features_mask: Optional[cp.ndarray] = None
    ) -> cp.ndarray:
        """Encode audio features.

        Args:
            input_features: (batch, mel_bins, time) mel spectrogram
            input_features_mask: (batch, time) mask for valid audio frames

        Returns:
            Audio embeddings (num_valid_frames, hidden_size) or (batch, seq, hidden) if no mask
        """
        audio_features = self.audio_encoder(input_features)
        projected = self.multi_modal_projector(audio_features)

        if input_features_mask is not None:
            # Compute audio lengths after conv layers (matching HF implementation)
            audio_lengths = cp.sum(input_features_mask, axis=-1)
            for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
                audio_lengths = (audio_lengths + 2 * padding - (kernel_size - 1) - 1) // stride + 1
            merge_factor = 4
            post_lengths = (audio_lengths - merge_factor) // merge_factor + 1

            # Create mask and extract valid embeddings
            seq_len = projected.shape[1]
            valid_mask = cp.arange(seq_len)[None, :] < post_lengths[:, None]
            # Flatten valid frames (removes batch dimension for now)
            projected = projected[valid_mask]

        return projected

    def decode(
        self,
        input_ids: Optional[cp.ndarray] = None,
        inputs_embeds: Optional[cp.ndarray] = None,
        attention_mask: Optional[cp.ndarray] = None,
        past_key_values: Optional[List[Tuple[cp.ndarray, cp.ndarray]]] = None,
        use_cache: bool = False
    ) -> Union[cp.ndarray, Tuple[cp.ndarray, List[Tuple[cp.ndarray, cp.ndarray]]]]:
        """Decode to logits with optional KV cache support."""
        result = self.text_decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        if use_cache:
            hidden_states, present_key_values = result
            logits = self.lm_head(hidden_states)
            return logits, present_key_values
        else:
            hidden_states = result
            logits = self.lm_head(hidden_states)
            return logits

    def forward(
        self,
        input_features: cp.ndarray,
        input_ids: Optional[cp.ndarray] = None
    ) -> cp.ndarray:
        """Full forward pass."""
        # Encode audio
        audio_embeds = self.encode_audio(input_features)

        if input_ids is not None:
            # Get text embeddings
            text_embeds = self.text_decoder.embed_tokens(input_ids)
            # Concatenate audio and text
            inputs_embeds = cp.concatenate([audio_embeds, text_embeds], axis=1)
        else:
            inputs_embeds = audio_embeds

        # Decode
        logits = self.decode(inputs_embeds=inputs_embeds)
        return logits

    def generate(
        self,
        input_features: cp.ndarray,
        input_ids: Optional[cp.ndarray] = None,
        input_features_mask: Optional[cp.ndarray] = None,
        attention_mask: Optional[cp.ndarray] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        audio_pad_token_id: int = 59260  # <|pad|> token for audio
    ) -> cp.ndarray:
        """Generate tokens from audio with proper chat template format.

        Args:
            input_features: (batch, mel_bins, time) mel spectrogram
            input_ids: (batch, seq_len) token IDs with audio placeholders (<|pad|>)
            input_features_mask: (batch, time) mask for valid audio frames
            attention_mask: (batch, seq_len) attention mask
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            audio_pad_token_id: Token ID for audio placeholders

        Returns:
            Generated token IDs
        """
        # Encode audio
        audio_embeds = self.encode_audio(input_features, input_features_mask)

        if input_ids is not None:
            batch_size = input_ids.shape[0]

            # Handle batch dimension for audio embeddings
            if audio_embeds.ndim == 3:
                # (batch, seq, hidden) -> (seq, hidden) for single batch
                audio_embeds = audio_embeds[0]

            # Get text embeddings for all tokens
            text_embeds = self.text_decoder.embed_tokens(input_ids)

            # Find audio placeholder positions and INSERT audio embeddings
            # Note: We INSERT all audio embeddings at the first pad position,
            # rather than replacing pad tokens 1-to-1
            audio_mask = (input_ids == audio_pad_token_id)
            audio_positions = cp.where(audio_mask[0])[0]

            if len(audio_positions) > 0:
                # Find the first and last pad positions
                first_pad_pos = int(audio_positions[0].get())
                last_pad_pos = int(audio_positions[-1].get())

                # Split text embeddings: before pads, after pads
                before_audio = text_embeds[0, :first_pad_pos, :]  # (first_pad_pos, hidden)
                after_audio = text_embeds[0, last_pad_pos + 1:, :]  # (remaining, hidden)

                # Concatenate: [before] + [audio_embeds] + [after]
                inputs_embeds = cp.concatenate([
                    before_audio[None, :, :],
                    audio_embeds[None, :, :],
                    after_audio[None, :, :]
                ], axis=1)
            else:
                # No pad tokens - just use text embeddings
                inputs_embeds = text_embeds

            # Track generated tokens (start from input_ids)
            generated = input_ids.copy()

        else:
            # No input_ids - use simple concatenation (legacy mode)
            batch_size = audio_embeds.shape[0] if audio_embeds.ndim == 3 else 1
            if audio_embeds.ndim == 2:
                audio_embeds = audio_embeds[None, :, :]
            inputs_embeds = audio_embeds
            generated = cp.full(
                (batch_size, 1),
                self.config.bos_token_id,
                dtype=cp.int64
            )

        # Track which sequences have finished (hit EOS)
        finished = cp.zeros(batch_size, dtype=cp.bool_)

        # Handle single or multiple EOS token IDs
        eos_token_ids = self.config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos_token_ids_cp = cp.array(eos_token_ids, dtype=cp.int64)

        # Autoregressive generation
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self.decode(inputs_embeds=inputs_embeds)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0 and top_k < next_token_logits.shape[-1]:
                top_k_indices = cp.argsort(next_token_logits, axis=-1)[:, -top_k:]
                top_k_logits = cp.take_along_axis(next_token_logits, top_k_indices, axis=-1)

                # Softmax
                top_k_logits_shifted = top_k_logits - cp.max(top_k_logits, axis=-1, keepdims=True)
                exp_logits = cp.exp(top_k_logits_shifted)
                probs = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)

                # Sample
                cumprobs = cp.cumsum(probs, axis=-1)
                samples = cp.random.uniform(size=(batch_size, 1))
                next_token_idx = cp.argmax(cumprobs >= samples, axis=-1)
                next_token = cp.take_along_axis(
                    top_k_indices,
                    next_token_idx[:, None],
                    axis=-1
                )
            else:
                next_token = cp.argmax(next_token_logits, axis=-1, keepdims=True)

            # Append to generated
            generated = cp.concatenate([generated, next_token], axis=1)

            # Check for EOS - mark sequences that generated any EOS token
            next_token_flat = next_token.flatten()
            is_eos = cp.any(next_token_flat[:, None] == eos_token_ids_cp[None, :], axis=1)
            finished = finished | is_eos

            # Stop if all sequences have finished
            if cp.all(finished):
                break

            # Update inputs_embeds with new token
            new_embeds = self.text_decoder.embed_tokens(next_token)
            inputs_embeds = cp.concatenate([inputs_embeds, new_embeds], axis=1)

        return generated

    def generate_v6(
        self,
        input_features: cp.ndarray,
        input_ids: Optional[cp.ndarray] = None,
        input_features_mask: Optional[cp.ndarray] = None,
        attention_mask: Optional[cp.ndarray] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        audio_pad_token_id: int = 59260
    ) -> cp.ndarray:
        """V6 optimized generate with pre-allocated buffers.

        Key optimizations:
        1. Pre-allocate token and embedding buffers
        2. Use buffer slicing instead of concatenation
        3. Reuse position IDs buffer
        """
        # Encode audio
        audio_embeds = self.encode_audio(input_features, input_features_mask)

        if input_ids is not None:
            batch_size = input_ids.shape[0]

            if audio_embeds.ndim == 3:
                audio_embeds = audio_embeds[0]

            text_embeds = self.text_decoder.embed_tokens(input_ids)

            audio_mask = (input_ids == audio_pad_token_id)
            audio_positions = cp.where(audio_mask[0])[0]

            if len(audio_positions) > 0:
                first_pad_pos = int(audio_positions[0].get())
                last_pad_pos = int(audio_positions[-1].get())

                before_audio = text_embeds[0, :first_pad_pos, :]
                after_audio = text_embeds[0, last_pad_pos + 1:, :]

                initial_embeds = cp.concatenate([
                    before_audio[None, :, :],
                    audio_embeds[None, :, :],
                    after_audio[None, :, :]
                ], axis=1)
            else:
                initial_embeds = text_embeds

            initial_len = initial_embeds.shape[1]
            generated = input_ids.copy()

        else:
            batch_size = audio_embeds.shape[0] if audio_embeds.ndim == 3 else 1
            if audio_embeds.ndim == 2:
                audio_embeds = audio_embeds[None, :, :]
            initial_embeds = audio_embeds
            initial_len = initial_embeds.shape[1]
            generated = cp.full((batch_size, 1), self.config.bos_token_id, dtype=cp.int64)

        # V6: Pre-allocate buffers
        max_total_len = initial_len + max_new_tokens
        hidden_size = self.config.text_hidden_size
        vocab_size = self.config.text_vocab_size

        # Pre-allocate embedding buffer
        embeds_buffer = cp.zeros((batch_size, max_total_len, hidden_size), dtype=cp.float32)
        embeds_buffer[:, :initial_len, :] = initial_embeds

        # Pre-allocate token buffer
        initial_tokens = generated.shape[1]
        tokens_buffer = cp.zeros((batch_size, initial_tokens + max_new_tokens), dtype=cp.int64)
        tokens_buffer[:, :initial_tokens] = generated

        current_len = initial_len
        current_tokens = initial_tokens

        finished = cp.zeros(batch_size, dtype=cp.bool_)

        eos_token_ids = self.config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos_token_ids_cp = cp.array(eos_token_ids, dtype=cp.int64)

        # V6: Pre-allocate top-k buffers if needed
        if top_k > 0:
            topk_indices_buf = cp.zeros((batch_size, top_k), dtype=cp.int64)
            topk_logits_buf = cp.zeros((batch_size, top_k), dtype=cp.float32)

        for step in range(max_new_tokens):
            # Use slice view instead of creating new array
            inputs_embeds = embeds_buffer[:, :current_len, :]

            # Get logits for next token
            logits = self.decode(inputs_embeds=inputs_embeds)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k sampling with pre-allocated buffers
            if top_k > 0 and top_k < vocab_size:
                # Get top-k indices
                top_k_indices = cp.argsort(next_token_logits, axis=-1)[:, -top_k:]
                top_k_logits = cp.take_along_axis(next_token_logits, top_k_indices, axis=-1)

                # Softmax
                top_k_logits_shifted = top_k_logits - cp.max(top_k_logits, axis=-1, keepdims=True)
                exp_logits = cp.exp(top_k_logits_shifted)
                probs = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)

                # Sample
                cumprobs = cp.cumsum(probs, axis=-1)
                samples = cp.random.uniform(size=(batch_size, 1))
                next_token_idx = cp.argmax(cumprobs >= samples, axis=-1)
                next_token = cp.take_along_axis(top_k_indices, next_token_idx[:, None], axis=-1)
            else:
                next_token = cp.argmax(next_token_logits, axis=-1, keepdims=True)

            # V6: Write to pre-allocated buffer instead of concatenate
            tokens_buffer[:, current_tokens] = next_token[:, 0]
            current_tokens += 1

            # Check for EOS
            next_token_flat = next_token.flatten()
            is_eos = cp.any(next_token_flat[:, None] == eos_token_ids_cp[None, :], axis=1)
            finished = finished | is_eos

            if cp.all(finished):
                break

            # V6: Write new embedding to buffer instead of concatenate
            new_embeds = self.text_decoder.embed_tokens(next_token)
            embeds_buffer[:, current_len, :] = new_embeds[:, 0, :]
            current_len += 1

        # Return only valid portion of buffer
        return tokens_buffer[:, :current_tokens]

    def generate_v8(
        self,
        input_features: cp.ndarray,
        input_ids: Optional[cp.ndarray] = None,
        input_features_mask: Optional[cp.ndarray] = None,
        attention_mask: Optional[cp.ndarray] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        audio_pad_token_id: int = 59260
    ) -> cp.ndarray:
        """V8 optimized generate with KV cache.

        Key optimization: Uses KV cache to avoid recomputing K/V for previous tokens.
        - Prefill: Process all initial embeddings, cache K/V
        - Decode: Only process new token, reuse cached K/V

        This reduces decode complexity from O(n²) to O(n) where n = sequence length.
        """
        # Encode audio
        audio_embeds = self.encode_audio(input_features, input_features_mask)

        if input_ids is not None:
            batch_size = input_ids.shape[0]

            if audio_embeds.ndim == 3:
                audio_embeds = audio_embeds[0]

            text_embeds = self.text_decoder.embed_tokens(input_ids)

            audio_mask = (input_ids == audio_pad_token_id)
            audio_positions = cp.where(audio_mask[0])[0]

            if len(audio_positions) > 0:
                first_pad_pos = int(audio_positions[0].get())
                last_pad_pos = int(audio_positions[-1].get())

                before_audio = text_embeds[0, :first_pad_pos, :]
                after_audio = text_embeds[0, last_pad_pos + 1:, :]

                initial_embeds = cp.concatenate([
                    before_audio[None, :, :],
                    audio_embeds[None, :, :],
                    after_audio[None, :, :]
                ], axis=1)
            else:
                initial_embeds = text_embeds

            generated = input_ids.copy()

        else:
            batch_size = audio_embeds.shape[0] if audio_embeds.ndim == 3 else 1
            if audio_embeds.ndim == 2:
                audio_embeds = audio_embeds[None, :, :]
            initial_embeds = audio_embeds
            generated = cp.full((batch_size, 1), self.config.bos_token_id, dtype=cp.int64)

        # V8: Prefill phase - process all initial embeddings and cache K/V
        logits, past_key_values = self.decode(
            inputs_embeds=initial_embeds,
            use_cache=True
        )
        next_token_logits = logits[:, -1, :] / temperature

        finished = cp.zeros(batch_size, dtype=cp.bool_)

        eos_token_ids = self.config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos_token_ids_cp = cp.array(eos_token_ids, dtype=cp.int64)

        vocab_size = self.config.text_vocab_size

        # Pre-allocate token buffer
        initial_tokens = generated.shape[1]
        tokens_buffer = cp.zeros((batch_size, initial_tokens + max_new_tokens), dtype=cp.int64)
        tokens_buffer[:, :initial_tokens] = generated
        current_tokens = initial_tokens

        for step in range(max_new_tokens):
            # Top-k sampling
            if top_k > 0 and top_k < vocab_size:
                top_k_indices = cp.argsort(next_token_logits, axis=-1)[:, -top_k:]
                top_k_logits = cp.take_along_axis(next_token_logits, top_k_indices, axis=-1)

                top_k_logits_shifted = top_k_logits - cp.max(top_k_logits, axis=-1, keepdims=True)
                exp_logits = cp.exp(top_k_logits_shifted)
                probs = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)

                cumprobs = cp.cumsum(probs, axis=-1)
                samples = cp.random.uniform(size=(batch_size, 1))
                next_token_idx = cp.argmax(cumprobs >= samples, axis=-1)
                next_token = cp.take_along_axis(top_k_indices, next_token_idx[:, None], axis=-1)
            else:
                next_token = cp.argmax(next_token_logits, axis=-1, keepdims=True)

            # Store token
            tokens_buffer[:, current_tokens] = next_token[:, 0]
            current_tokens += 1

            # Check for EOS
            next_token_flat = next_token.flatten()
            is_eos = cp.any(next_token_flat[:, None] == eos_token_ids_cp[None, :], axis=1)
            finished = finished | is_eos

            if cp.all(finished):
                break

            # V8: Decode phase - only process new token, reuse K/V cache
            new_embeds = self.text_decoder.embed_tokens(next_token)
            logits, past_key_values = self.decode(
                inputs_embeds=new_embeds,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = logits[:, -1, :] / temperature

        return tokens_buffer[:, :current_tokens]

    def generate_v8b(
        self,
        input_features: cp.ndarray,
        input_ids: Optional[cp.ndarray] = None,
        input_features_mask: Optional[cp.ndarray] = None,
        attention_mask: Optional[cp.ndarray] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        audio_pad_token_id: int = 59260,
        ignore_eos: bool = False  # For benchmarking: ignore EOS tokens
    ) -> cp.ndarray:
        """V8.1 optimized generate with pre-allocated KV buffers.

        Key optimization: Uses pre-allocated KV buffers to avoid concatenation.
        - Pre-allocates KV cache buffers for all layers
        - Uses slice assignment instead of concatenation
        """
        # Encode audio
        audio_embeds = self.encode_audio(input_features, input_features_mask)

        if input_ids is not None:
            batch_size = input_ids.shape[0]

            if audio_embeds.ndim == 3:
                audio_embeds = audio_embeds[0]

            text_embeds = self.text_decoder.embed_tokens(input_ids)

            audio_mask = (input_ids == audio_pad_token_id)
            audio_positions = cp.where(audio_mask[0])[0]

            if len(audio_positions) > 0:
                first_pad_pos = int(audio_positions[0].get())
                last_pad_pos = int(audio_positions[-1].get())

                before_audio = text_embeds[0, :first_pad_pos, :]
                after_audio = text_embeds[0, last_pad_pos + 1:, :]

                initial_embeds = cp.concatenate([
                    before_audio[None, :, :],
                    audio_embeds[None, :, :],
                    after_audio[None, :, :]
                ], axis=1)

                # Build proper token sequence for output (matching initial_embeds length)
                num_audio_embeds = audio_embeds.shape[0]
                before_tokens = input_ids[0, :first_pad_pos]
                after_tokens = input_ids[0, last_pad_pos + 1:]
                audio_placeholder_tokens = cp.full((num_audio_embeds,), audio_pad_token_id, dtype=cp.int64)
                generated = cp.concatenate([before_tokens, audio_placeholder_tokens, after_tokens])[None, :]
            else:
                initial_embeds = text_embeds
                generated = input_ids.copy()

        else:
            batch_size = audio_embeds.shape[0] if audio_embeds.ndim == 3 else 1
            if audio_embeds.ndim == 2:
                audio_embeds = audio_embeds[None, :, :]
            initial_embeds = audio_embeds
            generated = cp.full((batch_size, 1), self.config.bos_token_id, dtype=cp.int64)

        # V8.1: Pre-allocate KV buffers for all layers
        initial_len = initial_embeds.shape[1]
        max_seq_len = initial_len + max_new_tokens
        kv_buffers = self.text_decoder.allocate_kv_buffers(batch_size, max_seq_len)

        # Prefill phase - process all initial embeddings
        hidden_states, cache_pos = self.text_decoder.forward_with_kv_buffers(
            initial_embeds, kv_buffers, 0
        )
        logits = self.lm_head(hidden_states)
        next_token_logits = logits[:, -1, :] / temperature

        finished = cp.zeros(batch_size, dtype=cp.bool_)

        eos_token_ids = self.config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos_token_ids_cp = cp.array(eos_token_ids, dtype=cp.int64)

        vocab_size = self.config.text_vocab_size

        # Pre-allocate token buffer
        initial_tokens = generated.shape[1]
        tokens_buffer = cp.zeros((batch_size, initial_tokens + max_new_tokens), dtype=cp.int64)
        tokens_buffer[:, :initial_tokens] = generated
        current_tokens = initial_tokens

        for step in range(max_new_tokens):
            # Top-k sampling
            if top_k > 0 and top_k < vocab_size:
                top_k_indices = cp.argsort(next_token_logits, axis=-1)[:, -top_k:]
                top_k_logits = cp.take_along_axis(next_token_logits, top_k_indices, axis=-1)

                top_k_logits_shifted = top_k_logits - cp.max(top_k_logits, axis=-1, keepdims=True)
                exp_logits = cp.exp(top_k_logits_shifted)
                probs = exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)

                cumprobs = cp.cumsum(probs, axis=-1)
                samples = cp.random.uniform(size=(batch_size, 1))
                next_token_idx = cp.argmax(cumprobs >= samples, axis=-1)
                next_token = cp.take_along_axis(top_k_indices, next_token_idx[:, None], axis=-1)
            else:
                next_token = cp.argmax(next_token_logits, axis=-1, keepdims=True)

            # Store token
            tokens_buffer[:, current_tokens] = next_token[:, 0]
            current_tokens += 1

            # Check for EOS (skip if ignore_eos is True for benchmarking)
            if not ignore_eos:
                next_token_flat = next_token.flatten()
                is_eos = cp.any(next_token_flat[:, None] == eos_token_ids_cp[None, :], axis=1)
                finished = finished | is_eos

                if cp.all(finished):
                    break

            # V8.1: Decode phase - use pre-allocated KV buffers
            new_embeds = self.text_decoder.embed_tokens(next_token)
            hidden_states, cache_pos = self.text_decoder.forward_with_kv_buffers(
                new_embeds, kv_buffers, cache_pos
            )
            logits = self.lm_head(hidden_states)
            next_token_logits = logits[:, -1, :] / temperature

        return tokens_buffer[:, :current_tokens]
