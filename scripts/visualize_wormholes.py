# scripts/visualize_wormholes.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from cepler.models.transformer import CustomTransformer

def visualize_attention(model, seq_len=128, batch_size=1):
    model.eval()
    input_ids = torch.randint(0, 5000, (batch_size, seq_len), device='cuda', dtype=torch.long)
    with torch.no_grad():
        # Пропускаем через модель, но нам нужен доступ к промежуточным значениям
        # Для простоты возьмём первый блок и его лучи
        x = model.embedding(input_ids)
        x = x + model.pos_encoding[:, :seq_len, :]
        # Берём первый блок
        block = model.blocks[0]
        # Получаем лучи
        if hasattr(block.attn, 'ray_weights'):
            # Для статического или динамического RayAttention
            # Прокси-проход, чтобы вычислить лучи
            x_perm = x.transpose(1, 2)
            rays = torch.einsum('bcl, cr -> brl', x_perm, block.attn.ray_weights)
            # Усредняем по батчу и длине, чтобы увидеть распределение по лучам
            ray_importance = rays.mean(dim=(0, 2))  # (num_rays,)
            return ray_importance.cpu().numpy()
    return None

if __name__ == "__main__":
    # Создаём модель с wormhole-вниманием
    model = CustomTransformer(
        vocab_size=5000, d_model=256, num_heads=8, d_ff=512,
        num_layers=4, num_classes=2, max_len=128,
        attention_type='wormhole',  # новый тип
        num_rays=8,
        lightweight_ffn=True, ffn_rank=64,
        fuse_kernels=False
    ).cuda()

    # Визуализация для разных длин
    seq_lengths = [64, 128, 256, 512]
    for seq_len in seq_lengths:
        importance = visualize_attention(model, seq_len)
        if importance is not None:
            plt.plot(importance, label=f'seq_len={seq_len}', marker='o')
    plt.xlabel('Ray index')
    plt.ylabel('Average importance')
    plt.title('Распределение токенов по лучам (кротовым норам)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/wormhole_visualization.png', dpi=150)
    plt.show()