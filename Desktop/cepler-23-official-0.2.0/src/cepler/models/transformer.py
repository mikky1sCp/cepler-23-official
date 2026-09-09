# src/cepler/models/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from cepler.utils.quantized_linear import QuantizedLinear
from cepler.models.wormhole_attention import DynamicRayAttention

# ----------------------------------------------------------------------
# Кэш лучей (для генерации)
# ----------------------------------------------------------------------
class RayCache:
    def __init__(self):
        self.cached_rays = None

    def update(self, new_rays):
        if self.cached_rays is None:
            self.cached_rays = new_rays
        else:
            self.cached_rays = torch.cat([self.cached_rays, new_rays], dim=-1)
        return self.cached_rays

    def reset(self):
        self.cached_rays = None

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
# Вспомогательная функция для слияния ядер (RayAttention)
# ----------------------------------------------------------------------
def _fused_ray_attention(x_perm, ray_weights, sparse_rays=False, residual_echo=False):
    rays = torch.einsum('bcl, cr -> brl', x_perm, ray_weights)
    if sparse_rays:
        weights_ray = F.softmax(rays, dim=1)
        top_vals, top_idx = torch.topk(weights_ray, k=2, dim=1)
        mask_rays = torch.zeros_like(weights_ray).scatter_(1, top_idx, 1.0)
        rays = rays * mask_rays
    if residual_echo:
        rays = rays + rays.flip(-1) * 0.1
    attn_weights_len = F.softmax(rays, dim=2)
    weighted_sum = torch.einsum('brl,bcl->brc', attn_weights_len, x_perm)
    weights_ray = F.softmax(rays, dim=1)
    weights_ray_perm = weights_ray.permute(0, 2, 1)
    out = torch.einsum('bsr,brc->bsc', weights_ray_perm, weighted_sum)
    return out

if hasattr(torch, 'compile'):
    try:
        _fused_ray_attention = torch.compile(_fused_ray_attention, mode="max-autotune", fullgraph=True)
        print("RayAttention: torch.compile enabled with max-autotune")
    except Exception as e:
        print(f"RayAttention: torch.compile failed, using eager mode. Error: {e}")

# ----------------------------------------------------------------------
# RayAttention (статический, оригинальный)
# ----------------------------------------------------------------------
class RayAttention(nn.Module):
    def __init__(self, d_model, num_rays=8,
                 use_einsum=True, sparse_rays=False,
                 residual_echo=False, quantize_rays=False,
                 use_cache=False, use_matmul=False, fuse_kernels=False):
        super().__init__()
        self.num_rays = num_rays
        self.use_einsum = use_einsum
        self.sparse_rays = sparse_rays
        self.residual_echo = residual_echo
        self.quantize_rays = quantize_rays
        self.use_cache = use_cache
        self.use_matmul = use_matmul
        self.fuse_kernels = fuse_kernels

        self.ray_weights = nn.Parameter(torch.randn(d_model, num_rays) * 0.01)
        self.W_o = nn.Linear(d_model, d_model)
        if quantize_rays:
            self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.cache = None

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape

        if self.use_cache and self.cache is not None and seq_len == 1:
            x_perm = x.transpose(1, 2)
            if self.use_einsum and not self.use_matmul:
                new_rays = torch.einsum('bcl, cr -> brl', x_perm, self.ray_weights)
            else:
                new_rays = torch.matmul(x_perm.transpose(1,2), self.ray_weights).transpose(1,2)
            full_rays = self.cache.update(new_rays)
            rays = full_rays
        else:
            x_perm = x.transpose(1, 2)
            if self.use_einsum and not self.use_matmul:
                rays = torch.einsum('bcl, cr -> brl', x_perm, self.ray_weights)
            else:
                rays = torch.matmul(x_perm.transpose(1,2), self.ray_weights).transpose(1,2)
            if self.use_cache:
                if self.cache is None:
                    self.cache = RayCache()
                self.cache.cached_rays = rays

        if self.quantize_rays:
            w = self.ray_weights / (self.scale + 1e-8)
            w = torch.clamp(w, -1, 1)
            w_int = torch.round((w + 1) / 2 * 255)
            w_quant = (w_int / 255 * 2 - 1) * self.scale
            if self.use_einsum and not self.use_matmul:
                rays = torch.einsum('bcl, cr -> brl', x_perm, w_quant)
            else:
                rays = torch.matmul(x_perm.transpose(1,2), w_quant).transpose(1,2)

        if self.fuse_kernels and not self.use_cache:
            out = _fused_ray_attention(x_perm, self.ray_weights, self.sparse_rays, self.residual_echo)
        else:
            if self.sparse_rays:
                weights_ray = F.softmax(rays, dim=1)
                top_vals, top_idx = torch.topk(weights_ray, k=2, dim=1)
                mask_rays = torch.zeros_like(weights_ray).scatter_(1, top_idx, 1.0)
                rays = rays * mask_rays

            if self.residual_echo:
                rays = rays + rays.flip(-1) * 0.1

            attn_weights_len = F.softmax(rays, dim=2)
            weighted_sum = torch.einsum('brl,bcl->brc', attn_weights_len, x_perm)
            weights_ray = F.softmax(rays, dim=1)
            weights_ray_perm = weights_ray.permute(0, 2, 1)
            out = torch.einsum('bsr,brc->bsc', weights_ray_perm, weighted_sum)

        out = self.W_o(out)
        return out

# ----------------------------------------------------------------------
# Lightweight FFN (низкоранговая версия)
# ----------------------------------------------------------------------
class LightweightFFN(nn.Module):
    def __init__(self, d_model, d_ff, rank=None):
        super().__init__()
        if rank is None:
            rank = d_ff // 4
        self.fc1 = nn.Linear(d_model, rank)
        self.fc2 = nn.Linear(rank, d_ff)
        self.fc3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# ----------------------------------------------------------------------
# FeedForward (стандартный или квантизированный)
# ----------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, quantize=False, bits=4):
        super().__init__()
        if quantize:
            self.fc1 = QuantizedLinear(d_model, d_ff, bits=bits)
            self.fc2 = QuantizedLinear(d_ff, d_model, bits=bits)
        else:
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# ----------------------------------------------------------------------
# TransformerBlock
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff,
                 attention_type='multihead', num_rays=8,
                 use_einsum=True, sparse_rays=False,
                 residual_echo=False, quantize_rays=False,
                 quantize_ffn=False, ffn_bits=4,
                 use_cache=False, use_matmul=False, fuse_kernels=False,
                 lightweight_ffn=False, ffn_rank=None,
                 adaptive_rays=True):
        super().__init__()
        if attention_type == 'multihead':
            self.attn = MultiHeadAttention(d_model, num_heads)
        elif attention_type == 'ray':
            self.attn = RayAttention(d_model, num_rays,
                                     use_einsum, sparse_rays,
                                     residual_echo, quantize_rays,
                                     use_cache, use_matmul, fuse_kernels)
        elif attention_type == 'wormhole':
            self.attn = DynamicRayAttention(
                d_model, base_rays=num_rays, adaptive=adaptive_rays,
                use_einsum=use_einsum,
                sparse_rays=sparse_rays, residual_echo=residual_echo,
                quantize_rays=quantize_rays, use_cache=use_cache,
                use_matmul=use_matmul, fuse_kernels=fuse_kernels
            )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        if lightweight_ffn:
            self.ff = LightweightFFN(d_model, d_ff, rank=ffn_rank)
        else:
            self.ff = FeedForward(d_model, d_ff, quantize=quantize_ffn, bits=ffn_bits)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# ----------------------------------------------------------------------
# CustomTransformer
# ----------------------------------------------------------------------
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, num_heads=8, d_ff=512,
                 num_layers=6, num_classes=10, max_len=128,
                 attention_type='multihead', num_rays=8,
                 use_einsum=True, sparse_rays=False,
                 residual_echo=False, quantize_rays=False,
                 quantize_ffn=False, ffn_bits=4,
                 use_cache=False, use_matmul=False, fuse_kernels=False,
                 lightweight_ffn=False, ffn_rank=None,
                 num_rays_list=None, adaptive_rays=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer('pos_encoding', torch.zeros(1, max_len, d_model))
        self.num_layers = num_layers
        self.attention_type = attention_type
        self.use_cache = use_cache

        if num_rays_list is None:
            num_rays_list = [num_rays] * num_layers
        else:
            assert len(num_rays_list) == num_layers, "num_rays_list must have length num_layers"

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, attention_type,
                             num_rays_list[i], use_einsum, sparse_rays,
                             residual_echo, quantize_rays,
                             quantize_ffn, ffn_bits,
                             use_cache, use_matmul, fuse_kernels,
                             lightweight_ffn, ffn_rank,
                             adaptive_rays)
            for i in range(num_layers)
        ])

        self.classifiers = nn.ModuleList([
            nn.Linear(d_model, num_classes) for _ in range(num_layers)
        ])
        self.final_classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None, exit_threshold=None, return_all_logits=False):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]

        if self.use_cache:
            for block in self.blocks:
                if hasattr(block.attn, 'cache'):
                    block.attn.cache = None

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