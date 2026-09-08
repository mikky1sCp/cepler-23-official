# Пример замены MultiHeadAttention на RayAttention в существующей модели

import torch
import torch.nn as nn
from cepler.models.transformer import TransformerBlock

class MyModel(nn.Module):
    def __init__(self, use_ray=True):
        super().__init__()
        self.embed = nn.Embedding(5000, 256)
        self.block = TransformerBlock(
            d_model=256,
            num_heads=8,
            d_ff=512,
            attention_type='ray' if use_ray else 'multihead',
            num_rays=8,
            sparse_rays=True,
            residual_echo=True,
            quantize_rays=False,
        )
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.embed(x)
        x = self.block(x)
        x = x.mean(dim=1)
        return self.fc(x)

if __name__ == "__main__":
    model = MyModel(use_ray=True).cuda()
    x = torch.randint(0, 5000, (4, 128)).cuda()
    out = model(x)
    print(out.shape)  # torch.Size([4, 2])