"""
Generate time breakdown visualization for CuTile v10 decoder layer.
"""

import matplotlib.pyplot as plt
import numpy as np

# Component-wise breakdown from profiling (in milliseconds)
components = [
    ("input_layernorm", 0.031),
    ("QKV projection", 0.034),
    ("v_contiguous", 0.001),
    ("rope_get_cos_sin", 0.029),
    ("rope_apply", 0.032),
    ("kv_cache_update", 0.015),
    ("attention", 0.088),
    ("attn_reshape", 0.002),
    ("o_proj", 0.038),
    ("residual_add_attn", 0.009),
    ("post_attn_layernorm", 0.030),
    ("mlp", 0.254),
    ("residual_add_mlp", 0.009),
]

names = [c[0] for c in components]
times = [c[1] for c in components]
cumulative = np.cumsum(times)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Color coding by operation type
colors = []
for name in names:
    if 'norm' in name.lower():
        colors.append('#2ecc71')  # Green for normalization
    elif 'proj' in name.lower() or 'mlp' in name.lower():
        colors.append('#3498db')  # Blue for linear/MLP
    elif 'attention' in name.lower():
        colors.append('#e74c3c')  # Red for attention
    elif 'rope' in name.lower():
        colors.append('#9b59b6')  # Purple for RoPE
    elif 'residual' in name.lower():
        colors.append('#f39c12')  # Orange for residual
    else:
        colors.append('#95a5a6')  # Gray for others

# Plot 1: Bar chart with individual component times
x = np.arange(len(names))
bars = ax1.bar(x, times, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels on bars
for bar, time in zip(bars, times):
    height = bar.get_height()
    ax1.annotate(f'{time:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, rotation=45)

ax1.set_xlabel('Component', fontsize=12)
ax1.set_ylabel('Latency (ms)', fontsize=12)
ax1.set_title('CuTile v10 Decoder Layer - Component Latency Breakdown', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max(times) * 1.2)

# Add legend for color coding
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='Normalization'),
    Patch(facecolor='#3498db', label='Linear/MLP'),
    Patch(facecolor='#e74c3c', label='Attention'),
    Patch(facecolor='#9b59b6', label='RoPE'),
    Patch(facecolor='#f39c12', label='Residual'),
    Patch(facecolor='#95a5a6', label='Other'),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

# Plot 2: Cumulative line plot (step-by-step execution timeline)
ax2.plot(x, cumulative, 'b-o', linewidth=2, markersize=8, label='Cumulative time')
ax2.fill_between(x, cumulative, alpha=0.3)

# Add step markers with labels
for i, (name, cum_time) in enumerate(zip(names, cumulative)):
    ax2.annotate(f'{cum_time:.3f}ms',
                xy=(i, cum_time),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8, alpha=0.8)

# Add horizontal lines for key milestones
total_time = cumulative[-1]
ax2.axhline(y=total_time, color='red', linestyle='--', alpha=0.7, label=f'Total: {total_time:.3f}ms')
ax2.axhline(y=total_time/2, color='orange', linestyle=':', alpha=0.5, label=f'50%: {total_time/2:.3f}ms')

ax2.set_xlabel('Execution Step', fontsize=12)
ax2.set_ylabel('Cumulative Latency (ms)', fontsize=12)
ax2.set_title('CuTile v10 Decoder Layer - Cumulative Execution Timeline', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax2.grid(alpha=0.3)
ax2.legend(loc='upper left', fontsize=9)
ax2.set_ylim(0, total_time * 1.1)

plt.tight_layout()
plt.savefig('time_breakdown.png', dpi=150, bbox_inches='tight')
plt.savefig('time_breakdown.pdf', bbox_inches='tight')
print(f"Saved: time_breakdown.png and time_breakdown.pdf")

# Also print a text summary
print("\n" + "="*60)
print("CuTile v10 Decoder Layer Time Breakdown")
print("="*60)
print(f"{'Step':<5} {'Component':<20} {'Time (ms)':<12} {'Cumulative (ms)':<15} {'%Total':<8}")
print("-"*60)
for i, (name, time) in enumerate(components):
    pct = time / total_time * 100
    print(f"{i+1:<5} {name:<20} {time:<12.3f} {cumulative[i]:<15.3f} {pct:<8.1f}")
print("-"*60)
print(f"{'Total':<26} {total_time:<12.3f}")
print("="*60)

# Identify top 3 bottlenecks
sorted_components = sorted(components, key=lambda x: x[1], reverse=True)
print("\nTop 3 Bottlenecks:")
for i, (name, time) in enumerate(sorted_components[:3], 1):
    pct = time / total_time * 100
    print(f"  {i}. {name}: {time:.3f}ms ({pct:.1f}%)")
