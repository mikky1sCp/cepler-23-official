import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicRayAttention(nn.Module):
    """
    Внимание с динамическим числом лучей (кротовых нор).
    Число лучей адаптируется к длине последовательности: num_rays = max(4, int(sqrt(seq_len))).
    """
    def __init__(self, d_model, base_rays=8, adaptive=True,
                 use_einsum=True, sparse_rays=False, residual_echo=False,
                 quantize_rays=False, use_cache=False, use_matmul=False,
                 fuse_kernels=False):
        super().__init__()
        self.d_model = d_model
        self.base_rays = base_rays
        self.adaptive = adaptive
        self.use_einsum = use_einsum
        self.sparse_rays = sparse_rays
        self.residual_echo = residual_echo
        self.quantize_rays = quantize_rays
        self.use_cache = use_cache
        self.use_matmul = use_matmul
        self.fuse_kernels = fuse_kernels

        self.ray_weights = nn.Parameter(torch.randn(d_model, base_rays) * 0.01)
        self.W_o = nn.Linear(d_model, d_model)
        if quantize_rays:
            self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.cache = None

    def compute_num_rays(self, seq_len):
        if self.adaptive:
            num = int(math.sqrt(seq_len)) + 1
            num = max(4, min(num, self.base_rays * 2))
            return num
        else:
            return self.base_rays

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape
        num_rays = self.compute_num_rays(seq_len)

        # Адаптируем веса под новое число лучей
        if num_rays != self.ray_weights.size(1):
            new_weights = torch.zeros(d_model, num_rays, device=x.device)
            min_rays = min(num_rays, self.ray_weights.size(1))
            new_weights[:, :min_rays] = self.ray_weights[:, :min_rays]
            self.ray_weights = nn.Parameter(new_weights)

        x_perm = x.transpose(1, 2)  # (batch, d_model, seq_len)

        # Кэширование для генерации
        if self.use_cache and self.cache is not None and seq_len == 1:
            if self.use_einsum and not self.use_matmul:
                new_rays = torch.einsum('bcl, cr -> brl', x_perm, self.ray_weights)
            else:
                new_rays = torch.matmul(x_perm.transpose(1,2), self.ray_weights).transpose(1,2)
            full_rays = self.cache.update(new_rays)
            rays = full_rays
        else:
            if self.use_einsum and not self.use_matmul:
                rays = torch.einsum('bcl, cr -> brl', x_perm, self.ray_weights)
            else:
                rays = torch.matmul(x_perm.transpose(1,2), self.ray_weights).transpose(1,2)
            if self.use_cache:
                if self.cache is None:
                    from cepler.models.transformer import RayCache
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
            from cepler.models.transformer import _fused_ray_attention
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
