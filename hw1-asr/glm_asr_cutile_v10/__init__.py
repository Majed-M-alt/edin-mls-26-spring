"""
V10 - FP16 Optimized Pipeline
Target: Beat PyTorch baseline performance

Optimizations:
- cuBLAS FP16 GEMM for all Linear layers
- cuBLAS FP16 for MLP gate/up projections (1.56x faster than CuTile FP16)
- FlashAttention with native GQA

Usage:
    model.generate_v8b(...)
"""

import sys
import os

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from . import attention
attention.USE_FLASH_ATTENTION = True

from . import layers
# V10: Use cuBLAS FP16 for all operations
layers.Linear.BACKEND = 'cublas_fp16'  # cuBLAS FP16 for all sizes
layers.MLP.FUSED = True
layers.MLP.USE_FP16 = False  # Skip CuTile FP16 SwiGLU
layers.MLP.USE_CUBLAS_FP16 = True  # Use cuBLAS FP16 for gate/up (1.56x faster)
layers.EncoderMLP.FUSED = True  # Fused Linear+GELU for encoder

from . import model
from . import rope
from . import conv
from . import weight_loader
from . import flash_attention

DEFAULT_GENERATE = 'generate_v8b'
