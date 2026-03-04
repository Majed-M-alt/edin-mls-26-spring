import torch
import triton
import triton.language as tl
import time

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def test():
    print("Initializing tensors...")
    size = 9843200 # ~10MB
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output = torch.empty_like(x)

    print("Compiling and running kernel...")
    start = time.time()
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)
    torch.cuda.synchronize() # Wait for GPU to finish
    end = time.time()

    print(f"Success! Time taken: {(end - start)*1000:.2f} ms")
    print(f"Output check: {output[0]} (Should be non-zero)")

if __name__ == "__main__":
    test()