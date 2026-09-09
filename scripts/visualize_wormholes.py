# scripts/visualize_wormholes.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from cepler.models.transformer import CustomTransformer

def visualize_attention(model, seq_len=128, batch_size=1):
    model.eval()
    input_ids = torch.randint(0, 5000, (batch_size, seq_len), device='cuda', dtype=torch.long)
    with torch.no_grad():
        x = model.embedding(input_ids)
        x = x + model.pos_encoding[:, :seq_len, :]
        block = model.blocks[0]
        if hasattr(block.attn, 'ray_weights'):
            x_perm = x.transpose(1, 2)
            rays = torch.einsum('bcl, cr -> brl', x_perm, block.attn.ray_weights)
            ray_importance = rays.mean(dim=(0, 2))
            return ray_importance.cpu().numpy()
    return None

if __name__ == "__main__":
    model = CustomTransformer(
        vocab_size=5000, d_model=256, num_heads=8, d_ff=512,
        num_layers=4, num_classes=2, max_len=512,
        attention_type='wormhole',
        num_rays=8,
        lightweight_ffn=True, ffn_rank=64,
        fuse_kernels=False,
        adaptive_rays=True
    ).cuda()

    seq_lengths = [64, 128, 256, 512]
    for seq_len in seq_lengths:
        importance = visualize_attention(model, seq_len)
        if importance is not None:
            plt.plot(importance, label=f'seq_len={seq_len}', marker='o')
    plt.xlabel('Ray Index')
    plt.ylabel('Average Importance')
    plt.title('Token Distribution Across Rays (Wormholes)')
    plt.legend()
    plt.grid(True)
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/wormhole_visualization_en.png', dpi=150)
    plt.show()