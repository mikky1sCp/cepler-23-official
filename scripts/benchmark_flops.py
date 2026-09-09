import torch
from torch.utils.flop_counter import FlopCounterMode
from cepler.models.transformer import CustomTransformer

def count_flops(model, seq_len=512, batch_size=1):
    device = next(model.parameters()).device
    model.eval()
    input_ids = torch.randint(0, 5000, (batch_size, seq_len), device=device, dtype=torch.long)
    with torch.no_grad():
        flop_counter = FlopCounterMode(display=True)
        with flop_counter:
            _ = model(input_ids)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    return flop_counter.get_total_flops()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seq_len = 512
    batch_size = 1

    print("=" * 60)
    print("RAY ATTENTION + LIGHTWEIGHT FFN")
    ray_model = CustomTransformer(
        vocab_size=5000, d_model=256, num_heads=8, d_ff=512,
        num_layers=4, num_classes=2, max_len=seq_len,
        attention_type='ray', num_rays=8,
        lightweight_ffn=True, ffn_rank=64, fuse_kernels=False
    ).to(device)
    ray_flops = count_flops(ray_model, seq_len, batch_size)
    print(f"Ray total FLOPs: {ray_flops:.2e}")

    print("\n" + "=" * 60)
    print("MULTIHEAD ATTENTION + STANDARD FFN")
    mh_model = CustomTransformer(
        vocab_size=5000, d_model=256, num_heads=8, d_ff=512,
        num_layers=4, num_classes=2, max_len=seq_len,
        attention_type='multihead', lightweight_ffn=False, fuse_kernels=False
    ).to(device)
    mh_flops = count_flops(mh_model, seq_len, batch_size)
    print(f"Multihead total FLOPs: {mh_flops:.2e}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    if mh_flops > 0:
        reduction = (1 - ray_flops / mh_flops) * 100
        speedup = mh_flops / ray_flops
        print(f"FLOPs reduction: {reduction:.2f}%")
        print(f"Estimated speedup (by FLOPs): {speedup:.2f}x")