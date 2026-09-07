# Cepler-23: Energy-Efficient Transformer with Ray Attention

**Cepler-23** is an experimental transformer architecture that replaces quadratic self-attention with **Ray Attention**. Instead of pairwise interaction between all tokens, each token is projected onto a fixed number of directions (rays), resulting in linear complexity of **O(n × K)** rather than **O(n²)**.

## Key results

| Length | Model | Batch | Throughput (samples/s) | Energy per sample (Wh) | Speedup | Energy savings |
|--------|-------|-------|------------------------|------------------------|---------|----------------|
| 512    | Ray   | 16    | 334                    | 0.000049               | 2.52x   | 63.6%          |
| 512    | Multi | 16    | 129                    | 0.000131               |         |                |
| 2048   | Ray   | 8     | 84.4                   | 0.000203               | 6.26x   | 85.2%          |
| 2048   | Multi | 8     | 13.5                   | 0.001374               |         |                |

At a sequence length of **2048** tokens, RayAttention is **6.26 times faster** and consumes **85% less energy**.

## Features
- **Ray Attention** – projection onto K rays (default: 8).
- **Sparse Rays** – activation of only the top 2 rays (an additional 10–20% efficiency gain).
- **Residual Echo** – reflection for stabilization (virtually cost-free).
- **FP16** – mixed-precision support.
- **Early Exit** – capability to exit after any layer given high confidence.

## Installation

```bash
git clone https://github.com/mikky1sCp/Cepler-23-official.git
CD Cepler-23-official
pip install -r requirements.txt
```

## Quick Start
Running the benchmark with a length of 2048:
```bash
python -m scripts.benchmark --seq_len 2048 --batch_size 8 --fp16
```
Integration into your project:
```python
from cepler.models.transformer import CustomTransformer

model = CustomTransformer(
    vocab_size=10000,
    d_model=256,
    num_layers=6,
    attention_type='ray',       # 'ray' or 'multihead'
    num_rays=8,
    sparse_rays=True,           # optional
    residual_echo=True,         # optional
)
```
2. Replacing standard MultiHeadAttention with RayAttention
Let's say you have a model with standard self-attention:
```python
import torch.nn as nn
from torch.nn import MultiheadAttention

class MyModel(nn.Module):
    def __init__(self):
        self.attn = MultiheadAttention(embed_dim=256, num_heads=8)
```
3. Configuration for your specific task
Parameters that can be tweaked:
## ⚙️ RayAttention Parameters

| Parameter | Default Value | Description | Recommendation |
|----------|----------------------|----------|--------------|
| `num_rays` | 8 | Number of fixed directions (rays) | 8–16 for a balance of speed and quality |
| `sparse_rays` | `False` | Keeps only the top 2 rays | **Enable** to save ~30% energy |
| `residual_echo` | `False` | Adds a reflection (flip) for stability | Recommended when using sparsity |
| `use_einsum` | `True` | Uses `einsum` for projection (faster on GPU) | Keep as `True` on NVIDIA GPUs |
| `quantize_rays` | `False` | INT8 simulation for projection weights | Experimental; yields +10–15% savings |

4. Usage with Early Exit
```python
model = CustomTransformer(..., num_layers=6)
# During inference:
logits, exit_layer, confidence = model(input_ids, exit_threshold=0.95)
# If exit_layer != -1, the model exited early
```
5. Energy consumption measurement
```python
from cepler.utils.energy_monitor import EnergyMonitor

with EnergyMonitor() as mon:
    output = model(input_ids)
report = mon.stop()
print(f"Energy: {report['energy_wh']:.4f} Wh")
```
6. Deployment recommendations
- For long sequences (>512), use RayAttention to maximize savings.

- Enable `sparse_rays=True` and `residual_echo=True` for added efficiency.

- Use FP16 (autocast) throughout.

- Set `num_rays=8` for batch sizes > 16; for batches < 8, you can reduce this to 4.

- On a GTX 1660S, the optimal batch size for RayAttention is 64 with `seq_len=2048`.
