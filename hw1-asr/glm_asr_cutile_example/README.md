# V1 - Initial CuPy Implementation

## Performance: 3652ms (PyTorch: 232ms) - 15.7x slower

## Key Characteristics:
1. Pure CuPy tensor operations for compute
2. No FlashAttention (materializes full attention matrix)
3. Basic CuTile linear kernel with 16x16 tiles
4. Standard scaled_dot_product_attention using einsum

## Implementation Details:
- Uses CuPy's `einsum` for attention: `scores = cp.einsum('bnqd,bnkd->bnqk', q, k)`
- High memory bandwidth from large intermediate tensors
- Lack of kernel fusion optimizations
- No tensor core utilization

## Bottlenecks:
- Materializing full attention matrix (O(n^2) memory)
- Non-optimized attention computation
- Small tile sizes causing excessive kernel launch overhead

## Key Files:
- attention.py: `USE_FLASH_ATTENTION = False`
- layers.py: Basic CuTile linear kernel

## Usage:
```python
from versions.v1_initial_cupy import layers, model, attention
# or set attention.USE_FLASH_ATTENTION = False
```
