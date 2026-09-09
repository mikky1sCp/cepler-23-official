# scripts/benchmark_time.py
import torch
import time
from cepler.models.transformer import CustomTransformer
import matplotlib.pyplot as plt
import os
import numpy as np

def measure_time(model, seq_len, batch_size=1, num_warmup=5, num_iters=30):
    model.eval()
    input_ids = torch.randint(0, 5000, (batch_size, seq_len), device='cuda', dtype=torch.long)
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(input_ids)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters * 1000  # ms
    return elapsed

seq_lengths = [128, 256, 512, 1024, 2048]
attn_types = ['ray', 'multihead', 'wormhole']
results = {}

for attn_type in attn_types:
    print(f"\n=== Testing {attn_type} ===")
    times = []
    for seq_len in seq_lengths:
        print(f"  seq_len={seq_len}...")
        model = CustomTransformer(
            vocab_size=5000, d_model=256, num_heads=8, d_ff=512,
            num_layers=4, num_classes=2, max_len=seq_len,
            attention_type=attn_type,
            num_rays=8,
            lightweight_ffn=True if attn_type != 'multihead' else False,
            ffn_rank=64 if attn_type != 'multihead' else None,
            fuse_kernels=False,
            adaptive_rays=True if attn_type == 'wormhole' else False
        ).cuda()
        t = measure_time(model, seq_len)
        times.append(t)
        del model
        torch.cuda.empty_cache()
    results[attn_type] = times

# Plot
plt.figure(figsize=(12, 6))
colors = {'ray': 'blue', 'multihead': 'red', 'wormhole': 'green'}
markers = {'ray': 'o', 'multihead': 's', 'wormhole': '^'}
for attn_type in attn_types:
    plt.plot(seq_lengths, results[attn_type], marker=markers[attn_type],
             color=colors[attn_type], label=f'{attn_type}')

plt.xlabel('Sequence Length')
plt.ylabel('Inference Time (ms)')
plt.title('Inference Time Comparison (without torch.compile)')
plt.legend()
plt.grid(True)
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/time_vs_seqlen_en.png', dpi=150)
plt.show()

# Print table
print("\nResults (time in ms):")
print(f"{'seq_len':>8} ", end="")
for attn_type in attn_types:
    print(f"{attn_type:>14}", end="")
print()
for i, l in enumerate(seq_lengths):
    print(f"{l:>8} ", end="")
    for attn_type in attn_types:
        print(f"{results[attn_type][i]:>14.2f}", end="")
    print()

print("\nSpeedup vs MultiHead:")
print(f"{'seq_len':>8} ", end="")
for attn_type in ['ray', 'wormhole']:
    print(f"{attn_type:>14}", end="")
print()
for i, l in enumerate(seq_lengths):
    print(f"{l:>8} ", end="")
    base = results['multihead'][i]
    for attn_type in ['ray', 'wormhole']:
        speedup = base / results[attn_type][i] if results[attn_type][i] > 0 else np.inf
        print(f"{speedup:>13.2f}x", end="")
    print()