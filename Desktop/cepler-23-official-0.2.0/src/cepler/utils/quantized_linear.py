import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=4, initial_scale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.n_levels = 2 ** bits

        # Обучаемый масштаб (один на слой)
        self.scale = nn.Parameter(torch.ones(1) * initial_scale)

        # Веса в формате FP32 (будем квантизовать при forward)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Квантизация весов
        weight_scaled = self.weight / (self.scale + 1e-8)
        weight_clamped = torch.clamp(weight_scaled, -1, 1)
        weight_quant = torch.round((weight_clamped + 1) / 2 * (self.n_levels - 1))
        weight_dequant = (weight_quant / (self.n_levels - 1) * 2 - 1) * self.scale

        out = F.linear(x, weight_dequant, self.bias)
        return out