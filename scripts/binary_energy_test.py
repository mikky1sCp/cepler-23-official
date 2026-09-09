import torch
from torch.utils.data import DataLoader
from cepler.models.transformer import CustomTransformer
from cepler.utils.energy_monitor import EnergyMonitor
from cepler.utils.datasets import TextDataset
from sklearn.datasets import fetch_20newsgroups

def test_energy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    categories = ['comp.graphics', 'sci.space']
    newsgroups = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    texts = newsgroups.data[:300]
    labels = newsgroups.target[:300]

    dataset = TextDataset(texts, labels, vocab_size=5000, max_len=64)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = CustomTransformer(
        vocab_size=5000, d_model=256, num_heads=8, d_ff=512,
        num_layers=4, num_classes=2
    ).to(device)

    try:
        state_dict = torch.load('binary_transformer.pth', map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print("Loaded binary_transformer.pth")
    except FileNotFoundError:
        print("Warning: binary_transformer.pth not found, using random weights")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        print("Using random weights")

    model.eval()

    thresholds = [None, 0.9, 0.95, 0.99]
    results = []

    for thresh in thresholds:
        total_energy = 0.0
        total_samples = 0
        total_blocks = 0

        with EnergyMonitor() as mon:
            with torch.no_grad():
                for x, _ in loader:
                    x = x.to(device)
                    logits, exit_block, _ = model(x, exit_threshold=thresh)
                    batch_size = x.size(0)
                    if exit_block == -1:
                        total_blocks += batch_size * model.num_layers
                    else:
                        total_blocks += batch_size * (exit_block + 1)
                    total_samples += batch_size
            report = mon.stop()

        avg_energy = report['energy_wh'] / total_samples if total_samples > 0 else 0.0
        avg_blocks = total_blocks / total_samples if total_samples > 0 else 0.0
        results.append({
            'threshold': thresh,
            'total_energy_wh': report['energy_wh'],
            'avg_energy_wh': avg_energy,
            'elapsed_s': report['elapsed_sec'],
            'avg_power_w': report['avg_power_w'],
            'avg_blocks': avg_blocks,
        })
        print(f"Threshold: {thresh}, Total Energy: {report['energy_wh']:.4f} Wh, "
              f"Avg per sample: {avg_energy:.6f} Wh, Avg blocks: {avg_blocks:.2f}")

    print("\n=== Сравнение ===")
    base_energy = results[0]['avg_energy_wh']
    for r in results[1:]:
        saving = (1 - r['avg_energy_wh'] / base_energy) * 100 if base_energy > 0 else 0.0
        print(f"Threshold {r['threshold']}: экономия энергии {saving:.2f}%, "
              f"среднее число блоков {r['avg_blocks']:.2f} (из {model.num_layers})")

if __name__ == "__main__":
    test_energy()