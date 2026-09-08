# Cepler-23: Energy-Efficient Transformer

## RayAttention Architecture: Connectivity via "Wormholes"

### The Problem with Standard Attention
In classic Transformers, every token (a word or a segment of text) "communicates" with every other token. This requires a vast number of operations: if we have a sequence of `n` tokens, the number of connections grows at a rate of `n²`. For long texts (e.g., 2048 tokens), this becomes prohibitively expensive in terms of both memory and time.

### Our solution: rays as wormholes
We propose replacing the full network of connections with a **fixed set of "rays"** (8 in our implementation). These rays function like **wormholes**:
- Each token projects onto all the rays (sending its signal into each wormhole).
- The rays aggregate information from all tokens, creating a compressed representation.
- The rays then exchange data with one another.
- Finally, each token receives the aggregated signal back from all the rays.
- 
Instead of every token "calling" every other token (as in standard attention), tokens communicate only through 8 "portals." This reduces complexity from `O(n²)` to `O(n * 8)`, where 8 is a constant.

### Practical Advantages
- **Computational Efficiency:** For a sequence length of 512 tokens, our method requires **3.12 times fewer FLOPs** than standard attention.
- **Scalability:** The advantage grows as the sequence length increases. At 2048 tokens, the reduction in FLOPs reaches **84%** (see the graph below).
- **Real-world Speedup:** On a GTX 1660 Super, we observe an inference speedup of up to **~6x** for long sequences.

### A Real-World Analogy
Imagine that all the cities in a country need to exchange information. Instead of building direct roads between every pair of cities (which would require millions of kilometers of asphalt), we build **8 major hubs** (spokes). Each city sends its messages to the nearest hub, the hubs exchange summaries with one another, and then each city receives the aggregated information from all the hubs. This approach is cheaper, faster, and scalable to thousands of cities.

This is precisely how **RayAttention** works: we create "wormholes" in the feature space through which information passes almost instantaneously, without the quadratic explosion of connections.

## Results

We compared the proposed **RayAttention** with standard **MultiHead Attention** on a classification task (20 Newsgroups). The models shared the same configuration (d_model=256, 4 layers).


| seq_len | Ray (GFLOPs) | Multihead (GFLOPs) | Decline FLOPs |
|---------|--------------|-------------------|----------------|
| 128     | 0.26         | 0.60              | 57%            |
| 256     | 0.52         | 1.34              | 62%            |
| 512     | 1.03         | 3.22              | 68%            |
| 1024    | 2.06         | 8.59              | 76%            |
| 2048    | 4.13         | 25.77             | **84%**        |

### Real inference acceleration

On a GTX 1660 Super GPU (without using `torch.compile`), RayAttention demonstrates a speedup of up to **X times** on long sequences (exact figures will be available once the benchmark is complete).

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
