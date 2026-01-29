# V8.2 - Memory Copy Optimization (Current)

## Performance: 422ms (PyTorch: 298ms) - 42% slower, 4% faster than V8.1

## Key Optimizations:
1. FlashAttention handles GQA natively (skip KV expansion)
2. Optimized contiguity checks
3. Explicit V tensor contiguous conversion

## GQA Optimization:
```python
# Before (V8.1): Expand KV heads before attention
if num_kv_heads != num_heads:
    k = expand_kv(k, num_repeats)  # Memory copy
    v = expand_kv(v, num_repeats)  # Memory copy

# After (V8.2): FlashAttention handles GQA via QUERY_GROUP_SIZE
if USE_FLASH_ATTENTION and attention_mask is None:
    # Pass directly without expansion
    return flash_attention(q, k, v, scale, causal=is_causal)
```

## Contiguity Optimization:
```python
# V8.2: Only make contiguous when necessary
if not v.flags['C_CONTIGUOUS']:
    v = cp.ascontiguousarray(v)
```

## Changes from V8.1:
- attention.py: Skip KV expansion for FlashAttention path
- flash_attention.py: Added QUERY_GROUP_SIZE parameter for native GQA
- Reduced unnecessary memory copies in attention computation

## Remaining Bottlenecks:
1. cupy_copy (15.5%) - reshape/transpose overhead
2. gemvx cuBLAS (16.2%) - decode linear layers
3. FlashAttention padding - seq_q=1 pads to 64

## Profile Analysis:
| Operation | Time % | Notes |
|-----------|--------|-------|
| cupy_copy | 15.5% | Reshape/transpose |
| gemvx | 16.2% | cuBLAS GEMV |
| flash_attention | 12.8% | With padding overhead |
| linear_kernel_tf32 | 8.4% | Prefill matmuls |

## Key Files:
- attention.py: GQA-aware FlashAttention dispatch
- flash_attention.py: Native GQA support
- model.py: Complete generate_v8b() implementation
