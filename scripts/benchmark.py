import torch
import time
import argparse
from cepler.models.transformer import CustomTransformer
from cepler.utils.energy_monitor import EnergyMonitor

# Включаем оптимизации cuDNN
torch.backends.cudnn.benchmark = True

def load_model(attention_type, seq_len, **kwargs):
    model = CustomTransformer(
        vocab_size=5000,
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        num_classes=2,
        max_len=seq_len,
        attention_type=attention_type,
        num_rays=kwargs.get('num_rays', 8),
        use_einsum=kwargs.get('use_einsum', True),
        sparse_rays=kwargs.get('sparse_rays', False),
        residual_echo=kwargs.get('residual_echo', False),
        quantize_rays=kwargs.get('quantize_rays', False)
    ).cuda()
    try:
        state_dict = torch.load(f'{attention_type}_transformer.pth')
        state_dict.pop('pos_encoding', None)
        model.load_state_dict(state_dict, strict=False)
    except FileNotFoundError:
        print(f"Предупреждение: файл {attention_type}_transformer.pth не найден, используем случайные веса")
    return model

def measure(attention_type, seq_len=512, batch_size=8, num_batches=50, use_fp16=True, **kwargs):
    model = load_model(attention_type, seq_len, **kwargs)
    model.eval()
    x = torch.randint(0, 5000, (batch_size, seq_len)).cuda()
    with torch.no_grad():
        for _ in range(5):
            if use_fp16:
                with torch.amp.autocast('cuda'):
                    _ = model(x)
            else:
                _ = model(x)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_batches):
            if use_fp16:
                with torch.amp.autocast('cuda'):
                    _ = model(x)
            else:
                _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    samples = batch_size * num_batches
    throughput = samples / elapsed
    with EnergyMonitor() as mon:
        with torch.no_grad():
            for _ in range(num_batches):
                if use_fp16:
                    with torch.amp.autocast('cuda'):
                        _ = model(x)
                else:
                    _ = model(x)
        torch.cuda.synchronize()
        report = mon.stop()
    avg_energy = report['energy_wh'] / samples
    return throughput, avg_energy, report['energy_wh']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_batches', type=int, default=50)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--use_einsum', action='store_true', default=True)
    parser.add_argument('--sparse_rays', action='store_true', default=False)
    parser.add_argument('--residual_echo', action='store_true', default=False)
    parser.add_argument('--quantize_rays', action='store_true', default=False)
    args = parser.parse_args()

    print(f"Тестирование с seq_len={args.seq_len}, batch_size={args.batch_size}, FP16={args.fp16}")
    print(f"Опции: einsum={args.use_einsum}, sparse={args.sparse_rays}, echo={args.residual_echo}, quant={args.quantize_rays}")

    ray_kwargs = {
        'num_rays': 8,
        'use_einsum': args.use_einsum,
        'sparse_rays': args.sparse_rays,
        'residual_echo': args.residual_echo,
        'quantize_rays': args.quantize_rays
    }
    mh_kwargs = {}

    ray_th, ray_en, ray_total = measure('ray', seq_len=args.seq_len,
                                        batch_size=args.batch_size,
                                        num_batches=args.num_batches,
                                        use_fp16=args.fp16,
                                        **ray_kwargs)
    mh_th, mh_en, mh_total = measure('multihead', seq_len=args.seq_len,
                                     batch_size=args.batch_size,
                                     num_batches=args.num_batches,
                                     use_fp16=args.fp16,
                                     **mh_kwargs)

    print(f"\n=== RAY (seq_len={args.seq_len}) ===")
    print(f"Throughput: {ray_th:.2f} samples/sec")
    print(f"Total energy: {ray_total:.6f} Wh")
    print(f"Avg energy per sample: {ray_en:.6f} Wh")

    print(f"\n=== MULTIHEAD (seq_len={args.seq_len}) ===")
    print(f"Throughput: {mh_th:.2f} samples/sec")
    print(f"Total energy: {mh_total:.6f} Wh")
    print(f"Avg energy per sample: {mh_en:.6f} Wh")

    print("\n=== СРАВНЕНИЕ ===")
    print(f"Speedup (Ray/Multihead): {ray_th/mh_th:.2f}x")
    if mh_en > 0:
        print(f"Energy saving: {(1 - ray_en/mh_en)*100:.2f}%")
    else:
        print("Energy values too small to compute percentage.")