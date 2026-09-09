import torch
import time
import argparse
from cepler.models.transformer import CustomTransformer
from cepler.utils.energy_monitor import EnergyMonitor

torch.backends.cudnn.benchmark = True

# Универсальное устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(attention_type, seq_len, **kwargs):
    suffix = "_light" if kwargs.get('lightweight_ffn', False) else ""
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
        quantize_rays=kwargs.get('quantize_rays', False),
        quantize_ffn=kwargs.get('quantize_ffn', False),
        ffn_bits=kwargs.get('ffn_bits', 4),
        use_cache=kwargs.get('use_cache', False),
        use_matmul=kwargs.get('use_matmul', False),
        fuse_kernels=kwargs.get('fuse_kernels', False),
        lightweight_ffn=kwargs.get('lightweight_ffn', False),
        ffn_rank=kwargs.get('ffn_rank', None),
        num_rays_list=kwargs.get('num_rays_list', None)
    ).to(device)

    try:
        state_dict = torch.load(f'{attention_type}_transformer{suffix}.pth', map_location=device)
        # Удаляем pos_encoding, так как он пересоздаётся в модели
        state_dict.pop('pos_encoding', None)
        # Строгая загрузка — гарантирует полное совпадение параметров
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded weights from {attention_type}_transformer{suffix}.pth")
    except FileNotFoundError:
        print(f"Warning: {attention_type}_transformer{suffix}.pth not found, using random weights")
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Using random weights (architecture mismatch?)")
    return model

def measure(attention_type, seq_len=512, batch_size=8, num_batches=50,
            use_fp16=True, compile_model=False, channels_last=False,
            **kwargs):
    model_kwargs = kwargs.copy()
    model = load_model(attention_type, seq_len, **model_kwargs)
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    model.eval()

    x = torch.randint(0, 5000, (batch_size, seq_len)).to(device)
    if channels_last and device.type == 'cuda':
        x = x.to(memory_format=torch.channels_last)

    # Прогрев
    with torch.no_grad():
        for _ in range(5):
            if use_fp16 and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    _ = model(x)
            else:
                _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_batches):
            if use_fp16 and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    _ = model(x)
            else:
                _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3 if device.type == 'cuda' else 0.0

    with EnergyMonitor() as mon:
        with torch.no_grad():
            for _ in range(num_batches):
                if use_fp16 and device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        _ = model(x)
                else:
                    _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        report = mon.stop()

    samples = batch_size * num_batches
    throughput = samples / elapsed
    avg_energy = report['energy_wh'] / samples if samples > 0 else 0.0
    return throughput, avg_energy, report['energy_wh'], peak_mem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=None, help='Общий batch size (переопределяет специфические)')
    parser.add_argument('--batch_size_ray', type=int, default=32, help='Batch size для Ray')
    parser.add_argument('--batch_size_mh', type=int, default=8, help='Batch size для Multihead')
    parser.add_argument('--num_batches', type=int, default=50)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--channels_last', action='store_true', default=False)
    parser.add_argument('--use_einsum', action='store_true', default=True)
    parser.add_argument('--use_matmul', action='store_true', default=False)
    parser.add_argument('--sparse_rays', action='store_true', default=False)
    parser.add_argument('--residual_echo', action='store_true', default=False)
    parser.add_argument('--quantize_rays', action='store_true', default=False)
    parser.add_argument('--quantize_ffn', action='store_true', default=False)
    parser.add_argument('--ffn_bits', type=int, default=4)
    parser.add_argument('--use_cache', action='store_true', default=False)
    parser.add_argument('--dynamic_rays', action='store_true', default=False)
    parser.add_argument('--fuse', action='store_true', default=False, help='Включить слияние ядер (torch.compile)')
    parser.add_argument('--lightweight_ffn', action='store_true', default=False, help='Использовать LightweightFFN')
    parser.add_argument('--only_ray', action='store_true', default=False, help='Пропустить бенчмарк Multihead')
    args = parser.parse_args()

    if args.batch_size is not None:
        args.batch_size_ray = args.batch_size
        args.batch_size_mh = args.batch_size

    if args.use_matmul:
        args.use_einsum = False

    print(f"Testing with seq_len={args.seq_len}, FP16={args.fp16}")
    print(f"Batch sizes: Ray={args.batch_size_ray}, Multihead={args.batch_size_mh}")
    print(f"Options: compile={args.compile}, channels_last={args.channels_last}, "
          f"einsum={args.use_einsum}, matmul={args.use_matmul}, sparse={args.sparse_rays}, "
          f"echo={args.residual_echo}, quant_rays={args.quantize_rays}, quant_ffn={args.quantize_ffn}, "
          f"cache={args.use_cache}, dynamic_rays={args.dynamic_rays}, fuse={args.fuse}, "
          f"lightweight_ffn={args.lightweight_ffn}")

    # Базовые настройки для Ray
    ray_kwargs = {
        'num_rays': 8,
        'use_einsum': args.use_einsum,
        'sparse_rays': args.sparse_rays,
        'residual_echo': args.residual_echo,
        'quantize_rays': args.quantize_rays,
        'quantize_ffn': args.quantize_ffn,
        'ffn_bits': args.ffn_bits,
        'use_cache': args.use_cache,
        'use_matmul': args.use_matmul,
        'fuse_kernels': args.fuse,
        'lightweight_ffn': args.lightweight_ffn,
        'ffn_rank': 64 if args.lightweight_ffn else None,
    }
    if args.dynamic_rays:
        ray_kwargs['num_rays_list'] = [4, 6, 8, 8]
    else:
        ray_kwargs['num_rays_list'] = None

    mh_kwargs = {
        'quantize_ffn': args.quantize_ffn,
        'ffn_bits': args.ffn_bits,
        'use_cache': False,
        'use_matmul': False,
        'fuse_kernels': False,
        'lightweight_ffn': args.lightweight_ffn,
        'ffn_rank': 64 if args.lightweight_ffn else None,
    }

    # Измеряем Ray
    ray_th, ray_en, ray_total, ray_mem = measure('ray',
                                                 seq_len=args.seq_len,
                                                 batch_size=args.batch_size_ray,
                                                 num_batches=args.num_batches,
                                                 use_fp16=args.fp16,
                                                 compile_model=args.compile,
                                                 channels_last=args.channels_last,
                                                 **ray_kwargs)

    if not args.only_ray:
        mh_th, mh_en, mh_total, mh_mem = measure('multihead',
                                                 seq_len=args.seq_len,
                                                 batch_size=args.batch_size_mh,
                                                 num_batches=args.num_batches,
                                                 use_fp16=args.fp16,
                                                 compile_model=args.compile,
                                                 channels_last=args.channels_last,
                                                 **mh_kwargs)
    else:
        mh_th = mh_en = mh_total = mh_mem = 0.0

    print(f"\n=== RAY (seq_len={args.seq_len}, batch_size={args.batch_size_ray}) ===")
    print(f"Throughput: {ray_th:.2f} samples/sec")
    print(f"Total energy: {ray_total:.6f} Wh")
    print(f"Avg energy per sample: {ray_en:.6f} Wh")
    print(f"Peak memory: {ray_mem:.3f} GB")

    if not args.only_ray:
        print(f"\n=== MULTIHEAD (seq_len={args.seq_len}, batch_size={args.batch_size_mh}) ===")
        print(f"Throughput: {mh_th:.2f} samples/sec")
        print(f"Total energy: {mh_total:.6f} Wh")
        print(f"Avg energy per sample: {mh_en:.6f} Wh")
        print(f"Peak memory: {mh_mem:.3f} GB")

        print("\n=== COMPARISON ===")
        if mh_th > 0:
            print(f"Speedup (Ray/Multihead): {ray_th/mh_th:.2f}x")
        if mh_en > 0:
            print(f"Energy saving: {(1 - ray_en/mh_en)*100:.2f}%")
        if mh_mem > 0:
            print(f"Memory reduction: {(1 - ray_mem/mh_mem)*100:.2f}%")
    else:
        print("\n=== RAY ONLY (Multihead skipped) ===")