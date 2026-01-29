"""
PyTorch benchmark for comparison.
"""
import time
import torch
import torch.nn as nn
import numpy as np

# Configuration (matching GLM-ASR-Nano)
BATCH = 1
NUM_HEADS = 28
NUM_KV_HEADS = 4
SEQ_K = 100
HIDDEN_SIZE = 3584
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
INTERMEDIATE_SIZE = 18944

device = torch.device("cuda")


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = torch.nn.functional.silu(gate) * up
        return self.down_proj(hidden)


def benchmark_pytorch_mlp():
    print("\n=== PyTorch MLP Benchmark (M=1) ===")

    mlp = SimpleMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE).to(device).to(torch.float32)
    x = torch.randn(BATCH, 1, HIDDEN_SIZE, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(50):
        _ = mlp(x)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(200):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = mlp(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    print(f"PyTorch MLP FP32: {np.median(times[50:]):.4f}ms")

    # FP16
    mlp_fp16 = mlp.half()
    x_fp16 = x.half()

    for _ in range(50):
        _ = mlp_fp16(x_fp16)
    torch.cuda.synchronize()

    times = []
    for _ in range(200):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = mlp_fp16(x_fp16)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    print(f"PyTorch MLP FP16: {np.median(times[50:]):.4f}ms")


def benchmark_pytorch_linear():
    print("\n=== PyTorch Linear Benchmark (M=1) ===")

    linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False).to(device).to(torch.float32)
    x = torch.randn(BATCH, 1, HIDDEN_SIZE, device=device, dtype=torch.float32)

    for _ in range(50):
        _ = linear(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(200):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = linear(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    print(f"PyTorch Linear FP32: {np.median(times[50:]):.4f}ms")


def benchmark_pytorch_attention():
    print("\n=== PyTorch Attention Benchmark (seq_q=1) ===")

    q = torch.randn(BATCH, NUM_HEADS, 1, HEAD_DIM, device=device, dtype=torch.float32)
    k = torch.randn(BATCH, NUM_KV_HEADS, SEQ_K, HEAD_DIM, device=device, dtype=torch.float32)
    v = torch.randn(BATCH, NUM_KV_HEADS, SEQ_K, HEAD_DIM, device=device, dtype=torch.float32)

    # Expand KV for GQA
    repeats = NUM_HEADS // NUM_KV_HEADS
    k_exp = k.repeat_interleave(repeats, dim=1)
    v_exp = v.repeat_interleave(repeats, dim=1)

    def attn_fn():
        scale = 1.0 / (HEAD_DIM ** 0.5)
        scores = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v_exp)

    for _ in range(50):
        _ = attn_fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(200):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = attn_fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    print(f"PyTorch Attention (eager): {np.median(times[50:]):.4f}ms")

    # Try SDPA
    try:
        def sdpa_fn():
            k_exp_local = k.repeat_interleave(repeats, dim=1)
            v_exp_local = v.repeat_interleave(repeats, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(q, k_exp_local, v_exp_local)

        for _ in range(50):
            _ = sdpa_fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(200):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = sdpa_fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        print(f"PyTorch SDPA: {np.median(times[50:]):.4f}ms")
    except Exception as e:
        print(f"SDPA failed: {e}")


def benchmark_pytorch_rmsnorm():
    print("\n=== PyTorch RMSNorm Benchmark ===")

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return self.weight * x

    norm = RMSNorm(HIDDEN_SIZE).to(device)
    x = torch.randn(BATCH, 1, HIDDEN_SIZE, device=device, dtype=torch.float32)

    for _ in range(50):
        _ = norm(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(200):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = norm(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    print(f"PyTorch RMSNorm: {np.median(times[50:]):.4f}ms")


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Decode Performance Benchmark")
    print("=" * 60)

    benchmark_pytorch_attention()
    benchmark_pytorch_rmsnorm()
    benchmark_pytorch_mlp()
    benchmark_pytorch_linear()

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)
