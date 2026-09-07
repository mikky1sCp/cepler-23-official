import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------------------------------------------------
# Multi‑Head Attention (стандартный)
# ----------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.size()
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(out)

# ----------------------------------------------------------------------
# RayAttention – лучевое внимание с оптимизациями
# ----------------------------------------------------------------------
class RayAttention(nn.Module):
    def __init__(self, d_model, num_rays=8,
                 use_einsum=True, sparse_rays=False,
                 residual_echo=False, quantize_rays=False):
        super().__init__()
        self.num_rays = num_rays
        self.use_einsum = use_einsum
        self.sparse_rays = sparse_rays
        self.residual_echo = residual_echo
        self.quantize_rays = quantize_rays

        self.ray_weights = nn.Parameter(torch.randn(d_model, num_rays) * 0.01)
        self.W_o = nn.Linear(d_model, d_model)

        if quantize_rays:
            self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        batch, seq_len, d_model = x.shape
        x_perm = x.transpose(1, 2)  # (batch, d_model, seq_len)

        # ---------- проекция на лучи ----------
        if self.use_einsum:
            rays = torch.einsum('bcl, cr -> brl', x_perm, self.ray_weights)  # (b, num_rays, seq_len)
        else:
            # fallback – через линейный слой (медленнее)
            rays = torch.einsum('bcl, cr -> brl', x_perm, self.ray_weights)

        # ---------- квантизация весов проекции (INT8 симуляция) ----------
        if self.quantize_rays:
            w = self.ray_weights / (self.scale + 1e-8)
            w = torch.clamp(w, -1, 1)
            w_int = torch.round((w + 1) / 2 * 255)
            w_quant = (w_int / 255 * 2 - 1) * self.scale
            rays = torch.einsum('bcl, cr -> brl', x_perm, w_quant)

        # ---------- разреженность (top‑k лучей) ----------
        if self.sparse_rays:
            # softmax по лучам для каждого токена
            weights_ray = F.softmax(rays, dim=1)          # (b, r, l)
            top_vals, top_idx = torch.topk(weights_ray, k=2, dim=1)
            mask_rays = torch.zeros_like(weights_ray).scatter_(1, top_idx, 1.0)
            rays = rays * mask_rays

        # ---------- отражение (residual echo) ----------
        if self.residual_echo:
            rays = rays + rays.flip(-1) * 0.1   # flip по длине (последняя ось)

        # ---------- агрегация по лучам ----------
        # 1) веса по длине (для каждого луча) – softmax по токенам
        attn_weights_len = F.softmax(rays, dim=2)          # (b, r, l)
        # взвешенное среднее по токенам → (b, r, d_model)
        weighted_sum = torch.einsum('brl,bcl->brc', attn_weights_len, x_perm)

        # 2) веса по лучам (для каждого токена) – softmax по лучам
        if self.sparse_rays:
            # используем уже вычисленные weights_ray (top‑2 softmax)
            weights_ray = F.softmax(rays, dim=1)          # (b, r, l)
        else:
            weights_ray = F.softmax(rays, dim=1)          # (b, r, l)

        # переставляем для удобства: (b, seq_len, num_rays)
        weights_ray_perm = weights_ray.permute(0, 2, 1)    # (b, l, r)

        # комбинируем: out_i = sum_r weights_ray[i,r] * weighted_sum[r]
        out = torch.einsum('bsr,brc->bsc', weights_ray_perm, weighted_sum)  # (b, l, d_model)

        out = self.W_o(out)
        return out

# ----------------------------------------------------------------------
# FFN, TransformerBlock, CustomTransformer – без изменений
# ----------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff,
                 attention_type='multihead', num_rays=8,
                 use_einsum=True, sparse_rays=False,
                 residual_echo=False, quantize_rays=False):
        super().__init__()
        if attention_type == 'multihead':
            self.attn = MultiHeadAttention(d_model, num_heads)
        elif attention_type == 'ray':
            self.attn = RayAttention(d_model, num_rays,
                                     use_einsum, sparse_rays,
                                     residual_echo, quantize_rays)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, num_heads=8, d_ff=512,
                 num_layers=6, num_classes=10, max_len=128,
                 attention_type='multihead', num_rays=8,
                 use_einsum=True, sparse_rays=False,
                 residual_echo=False, quantize_rays=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer('pos_encoding', torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, attention_type,
                             num_rays, use_einsum, sparse_rays,
                             residual_echo, quantize_rays)
            for _ in range(num_layers)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(d_model, num_classes) for _ in range(num_layers)
        ])
        self.final_classifier = nn.Linear(d_model, num_classes)
        self.num_layers = num_layers
        self.attention_type = attention_type

    def forward(self, x, mask=None, exit_threshold=None, return_all_logits=False):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        all_logits = []
        for idx, block in enumerate(self.blocks):
            x = block(x, mask)
            pooled = x.mean(dim=1)
            logits = self.classifiers[idx](pooled)
            all_logits.append(logits)
            if exit_threshold is not None:
                probs = F.softmax(logits, dim=1)
                max_prob, _ = torch.max(probs, dim=1)
                if (max_prob >= exit_threshold).all():
                    if return_all_logits:
                        return all_logits, idx
                    else:
                        return logits, idx, max_prob
        pooled = x.mean(dim=1)
        final_logits = self.final_classifier(pooled)
        all_logits.append(final_logits)
        if return_all_logits:
            return all_logits, -1
        else:
            return final_logits, -1, None