import torch
from torch.utils.flop_counter import FlopCounterMode
from cepler.models.transformer import CustomTransformer
import matplotlib.pyplot as plt
import os

def count_flops(model, seq_len, batch_size=1):
    model.eval()
    input_ids = torch.randint(0, 5000, (batch_size, seq_len), device='cuda', dtype=torch.long)
    with torch.no_grad():
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            _ = model(input_ids)
            torch.cuda.synchronize()
    return flop_counter.get_total_flops()

seq_lengths = [128, 256, 512, 1024, 2048]
attn_types = ['ray', 'multihead', 'wormhole']
results = {}

for attn_type in attn_types:
    print(f"\n=== Measuring FLOPs for {attn_type} ===")
    flops_list = []
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
        f = count_flops(model, seq_len)
        flops_list.append(f)
        del model
        torch.cuda.empty_cache()
    results[attn_type] = flops_list

# Plot
plt.figure(figsize=(12, 6))
colors = {'ray': 'blue', 'multihead': 'red', 'wormhole': 'green'}
markers = {'ray': 'o', 'multihead': 's', 'wormhole': '^'}
for attn_type in attn_types:
    plt.plot(seq_lengths, [f / 1e9 for f in results[attn_type]],
             marker=markers[attn_type], color=colors[attn_type], label=f'{attn_type}')

plt.xlabel('Sequence Length')
plt.ylabel('FLOPs (Giga)')
plt.title('Computational Complexity (FLOPs)')
plt.legend()
plt.grid(True)
plt.yscale('log')
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/flops_vs_seqlen_en.png', dpi=150)
plt.show()

# Print table
print("\nResults (GFLOPs):")
print(f"{'seq_len':>8} ", end="")
for attn_type in attn_types:
    print(f"{attn_type:>14}", end="")
print()
for i, l in enumerate(seq_lengths):
    print(f"{l:>8} ", end="")
    for attn_type in attn_types:
        print(f"{results[attn_type][i] / 1e9:>14.2f}", end="")
    print()

print("\nFLOPs reduction vs MultiHead (%):")
print(f"{'seq_len':>8} ", end="")
for attn_type in ['ray', 'wormhole']:
    print(f"{attn_type:>14}", end="")
print()
for i, l in enumerate(seq_lengths):
    print(f"{l:>8} ", end="")
    base = results['multihead'][i]
    for attn_type in ['ray', 'wormhole']:
        reduction = (1 - results[attn_type][i] / base) * 100 if base > 0 else 0
        print(f"{reduction:>13.2f}%", end="")
    print()
