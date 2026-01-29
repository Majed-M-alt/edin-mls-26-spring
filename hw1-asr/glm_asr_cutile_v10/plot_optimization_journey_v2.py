"""
Optimization journey visualization V2.
Shows two benchmark phases with proper context.
"""

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Data from README and profiling
# =============================================================================

# Phase 1: Original benchmark (prefill mode, no KV cache)
# These are relative measurements showing optimization effectiveness
phase1_versions = ["V1", "V2", "V3", "V4", "V5", "V6"]
phase1_times = [3652, 940, 364, 307, 300, 295]  # ms
phase1_optimizations = [
    "Initial CuPy",
    "FlashAttention",
    "cuBLAS Linear",
    "TF32 Tensor Core",
    "Fused Kernels",
    "Pre-alloc Buffers"
]

# Phase 2: Decode mode benchmark with KV cache (realistic ASR scenario)
# V10 measured: 0.585ms/layer, PyTorch: 0.697ms/layer (28 layers)
phase2_versions = ["V8", "V8.1", "V8.2", "V9", "V10", "PyTorch"]
phase2_times = [0.80, 0.78, 0.75, 0.81, 0.585, 0.697]  # ms per layer
phase2_optimizations = [
    "KV Cache",
    "Pre-alloc KV",
    "Memory Opt",
    "Optimized FP32",
    "FP16 + Fused QKV",
    "Baseline"
]

# =============================================================================
# Create visualization
# =============================================================================

fig = plt.figure(figsize=(16, 10))

# -----------------------------------------------------------------------------
# Plot 1: Phase 1 optimization (top-left)
# -----------------------------------------------------------------------------
ax1 = fig.add_subplot(2, 2, 1)

x1 = np.arange(len(phase1_versions))
colors1 = plt.cm.Reds(np.linspace(0.3, 0.9, len(phase1_versions)))[::-1]
bars1 = ax1.bar(x1, phase1_times, color=colors1, edgecolor='black', linewidth=0.5)

for bar, time, opt in zip(bars1, phase1_times, phase1_optimizations):
    height = bar.get_height()
    ax1.annotate(f'{time}ms\n{opt}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('Version', fontsize=11)
ax1.set_ylabel('Execution Time (ms)', fontsize=11)
ax1.set_title('Phase 1: Core Optimizations (Prefill Mode)', fontsize=13, fontweight='bold')
ax1.set_xticks(x1)
ax1.set_xticklabels(phase1_versions, fontsize=10)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)

# Speedup annotation
ax1.annotate(f'12.4x\nfaster', xy=(5, 295), xytext=(5.3, 1000),
            fontsize=12, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green'))

# -----------------------------------------------------------------------------
# Plot 2: Phase 1 step-by-step improvement (top-right)
# -----------------------------------------------------------------------------
ax2 = fig.add_subplot(2, 2, 2)

step_speedups1 = [1.0]
for i in range(1, len(phase1_times)):
    step_speedups1.append(phase1_times[i-1] / phase1_times[i])

colors_step = ['#27ae60' if s > 1.1 else '#3498db' if s > 1.0 else '#e74c3c' for s in step_speedups1]
bars2 = ax2.bar(x1, step_speedups1, color=colors_step, edgecolor='black', linewidth=0.5)

for bar, speedup, opt in zip(bars2, step_speedups1, phase1_optimizations):
    height = bar.get_height()
    label = f'{speedup:.2f}x' if speedup != 1.0 else 'base'
    ax2.annotate(f'{label}\n{opt}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8,
                color='green' if speedup > 1.1 else 'black')

ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
ax2.set_xlabel('Version', fontsize=11)
ax2.set_ylabel('Speedup vs Previous', fontsize=11)
ax2.set_title('Phase 1: Step-by-Step Improvement', fontsize=13, fontweight='bold')
ax2.set_xticks(x1)
ax2.set_xticklabels(phase1_versions, fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, max(step_speedups1) * 1.3)

# -----------------------------------------------------------------------------
# Plot 3: Phase 2 decode mode (bottom-left)
# -----------------------------------------------------------------------------
ax3 = fig.add_subplot(2, 2, 3)

x2 = np.arange(len(phase2_versions))
colors2 = ['#3498db'] * (len(phase2_versions) - 1) + ['#9b59b6']  # PyTorch is purple
colors2[4] = '#2ecc71'  # V10 is green (best)

bars3 = ax3.bar(x2, phase2_times, color=colors2, edgecolor='black', linewidth=0.5)

# PyTorch baseline line
pytorch_time = phase2_times[-1]
ax3.axhline(y=pytorch_time, color='purple', linestyle='--', linewidth=2,
            label=f'PyTorch baseline ({pytorch_time}ms)')

for bar, time, opt in zip(bars3, phase2_times, phase2_optimizations):
    height = bar.get_height()
    color = 'green' if time < pytorch_time else 'black'
    ax3.annotate(f'{time:.3f}ms\n{opt}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, color=color)

ax3.set_xlabel('Version', fontsize=11)
ax3.set_ylabel('Latency per Layer (ms)', fontsize=11)
ax3.set_title('Phase 2: Decode Mode with KV Cache (Real ASR Config)', fontsize=13, fontweight='bold')
ax3.set_xticks(x2)
ax3.set_xticklabels(phase2_versions, fontsize=10)
ax3.grid(axis='y', alpha=0.3)
ax3.legend(loc='upper right')

# Victory annotation
ax3.annotate('✓ 19% faster\nthan PyTorch!', xy=(4, 0.585), xytext=(4.5, 0.65),
            fontsize=11, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green'))

# -----------------------------------------------------------------------------
# Plot 4: Full journey line plot (bottom-right)
# -----------------------------------------------------------------------------
ax4 = fig.add_subplot(2, 2, 4)

# Normalize Phase 1 to per-layer time (assuming 28 layers and ~3000 tokens generated)
# This is an approximation for visualization
phase1_normalized = [t / 28 / 5 for t in phase1_times]  # rough normalization

# Combine for line plot
all_versions = phase1_versions + ["V7"] + phase2_versions[:-1]  # exclude PyTorch
all_times = phase1_normalized + [0.85] + phase2_times[:-1]  # V7 estimated

x_all = np.arange(len(all_versions))

# Plot CuTile journey
ax4.plot(x_all, all_times, 'b-o', linewidth=2, markersize=8, label='CuTile')

# Mark phase boundaries
ax4.axvline(x=5.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax4.annotate('Phase 1\n(Prefill)', xy=(2.5, max(all_times)*0.9), ha='center', fontsize=9, color='gray')
ax4.annotate('Phase 2\n(Decode)', xy=(9, max(all_times)*0.9), ha='center', fontsize=9, color='gray')

# PyTorch reference for decode phase
pytorch_line_x = [6, len(all_versions)-1]
pytorch_line_y = [pytorch_time, pytorch_time]
ax4.plot([6, len(all_versions)-1], [pytorch_time, pytorch_time], 'purple',
         linestyle='--', linewidth=2, label=f'PyTorch ({pytorch_time}ms)')

# Fill areas
for i in range(6, len(all_times)):
    if all_times[i] <= pytorch_time:
        ax4.fill_between([i-0.5, i+0.5], [all_times[i], all_times[i]],
                        [pytorch_time, pytorch_time], alpha=0.3, color='green')

ax4.set_xlabel('Version', fontsize=11)
ax4.set_ylabel('Latency per Layer (ms, normalized)', fontsize=11)
ax4.set_title('Complete Optimization Journey: V1 → V10', fontsize=13, fontweight='bold')
ax4.set_xticks(x_all)
ax4.set_xticklabels(all_versions, fontsize=9, rotation=45)
ax4.grid(alpha=0.3)
ax4.legend(loc='upper right')

plt.tight_layout()
plt.savefig('optimization_journey_v2.png', dpi=150, bbox_inches='tight')
plt.savefig('optimization_journey_v2.pdf', bbox_inches='tight')
print("Saved: optimization_journey_v2.png and optimization_journey_v2.pdf")

# =============================================================================
# Print summary
# =============================================================================
print("\n" + "="*70)
print("CuTile Optimization Journey Summary")
print("="*70)

print("\n📊 Phase 1: Core Optimizations (Prefill Mode)")
print("-"*50)
print(f"{'Version':<10} {'Time (ms)':<12} {'Speedup':<12} {'Optimization'}")
for i, (v, t, o) in enumerate(zip(phase1_versions, phase1_times, phase1_optimizations)):
    speedup = phase1_times[0] / t
    print(f"{v:<10} {t:<12} {speedup:<12.2f}x {o}")

print(f"\n✅ Total Phase 1 improvement: {phase1_times[0]/phase1_times[-1]:.1f}x faster")

print("\n📊 Phase 2: Decode Mode with KV Cache")
print("-"*50)
print(f"{'Version':<10} {'Time (ms)':<12} {'vs PyTorch':<12} {'Optimization'}")
for v, t, o in zip(phase2_versions, phase2_times, phase2_optimizations):
    if v != "PyTorch":
        ratio = pytorch_time / t
        status = "faster" if ratio > 1 else "slower"
        print(f"{v:<10} {t:<12.3f} {ratio:.2f}x {status:<6} {o}")
    else:
        print(f"{v:<10} {t:<12.3f} {'baseline':<12} {o}")

print(f"\n🎯 V10 beats PyTorch by {(1 - phase2_times[4]/pytorch_time)*100:.1f}%!")
print("="*70)
