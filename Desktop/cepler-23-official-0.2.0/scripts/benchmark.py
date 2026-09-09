import torch
import time
import argparse
from cepler.models.transformer import CustomTransformer
from cepler.utils.energy_monitor import EnergyMonitor

torch.backends.cudnn.benchmark = True

def load_model(attention_type, seq_len, **kwargs):
    # Определяем суффикс для файла весов (если используется LightweightFFN)
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
    ).cuda()
    try:
        state_dict = torch.load(f'{attention_type}_transformer{suffix}.pth')
        state_dict.pop('pos_encoding', None)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {attention_type}_transformer{suffix}.pth")
    except FileNotFoundError:
        print(f"Warning: {attention_type}_transformer{suffix}.pth not found, using random weights")
    return model

def measure(attention_type, seq_len=512, batch_size=8, num_batches=50,
            use_fp16=True, compile_model=False, channels_last=False,
            **kwargs):
    # Все параметры модели собираем в единый словарь
    model_kwargs = kwargs.copy()  # включает use_matmul, fuse_kernels, lightweight_ffn, ffn_rank и другие
    model = load_model(attention_type, seq_len, **model_kwargs)
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    model.eval()

    x = torch.randint(0, 5000, (batch_size, seq_len)).cuda()
    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    # Прогрев
    with torch.no_grad():
        for _ in range(5):
            if use_fp16:
                with torch.amp.autocast('cuda'):
                    _ = model(x)
            else:
                _ = model(x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
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
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

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

    samples = batch_size * num_batches
    throughput = samples / elapsed
    avg_energy = report['energy_wh'] / samples
    return throughput, avg_energy, report['energy_wh'], peak_mem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=None, help='Common batch size (overrides specific)')
    parser.add_argument('--batch_size_ray', type=int, default=32, help='Batch size for Ray')
    parser.add_argument('--batch_size_mh', type=int, default=8, help='Batch size for Multihead')
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
    parser.add_argument('--fuse', action='store_true', default=False, help='Enable kernel fusion (torch.compile)')
    parser.add_argument('--lightweight_ffn', action='store_true', default=False, help='Use LightweightFFN instead of standard FFN')
    parser.add_argument('--only_ray', action='store_true', default=False, help='Skip Multihead benchmark')
    args = parser.parse_args()

    # Если задан общий batch_size, переопределяем специфические
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

    # Базовые настройки для Ray (только параметры, специфичные для модели)
    ray_kwargs = {
        'num_rays': 8,
        'use_einsum': args.use_einsum,
        'sparse_rays': args.sparse_rays,
        'residual_echo': args.residual_echo,
        'quantize_rays': args.quantize_rays,
        'quantize_ffn': args.quantize_ffn,
        'ffn_bits': args.ffn_bits,
        'use_cache': args.use_cache,
        'use_matmul': args.use_matmul,   # теперь в kwargs
        'fuse_kernels': args.fuse,       # теперь в kwargs
        'lightweight_ffn': args.lightweight_ffn,
        'ffn_rank': 64 if args.lightweight_ffn else None,
    }
    if args.dynamic_rays:
        ray_kwargs['num_rays_list'] = [4, 6, 8, 8]
    else:
        ray_kwargs['num_rays_list'] = None

    # Для Multihead кэш и квантизация лучей не применяются
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

    # Измеряем Multihead (если не указан --only_ray)
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
        # Заглушка
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
        else:
            print("Energy values too small to compute percentage.")
        if mh_mem > 0:
            print(f"Memory reduction: {(1 - ray_mem/mh_mem)*100:.2f}% (Ray uses {ray_mem:.2f} GB, Multihead uses {mh_mem:.2f} GB)")
    else:
        print("\n=== RAY ONLY (Multihead skipped) ===")
        print(f"Throughput: {ray_th:.2f} samples/sec")
        print(f"Avg energy per sample: {ray_en:.6f} Wh")
        print(f"Peak memory: {ray_mem:.3f} GB")