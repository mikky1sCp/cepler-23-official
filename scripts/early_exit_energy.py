import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from cepler.models.transformer import CustomTransformer
from cepler.utils.energy_monitor import EnergyMonitor
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------------------------------------------------
# Датасет (дублируем для самодостаточности)
# ----------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab_size=5000, max_len=64):
        self.labels = labels
        self.max_len = max_len
        self.vocab_size = vocab_size
        vectorizer = CountVectorizer(max_features=vocab_size, lowercase=True)
        X = vectorizer.fit_transform(texts).toarray()
        X = X[:, :max_len]
        pad_width = max_len - X.shape[1]
        if pad_width > 0:
            X = np.pad(X, ((0, 0), (0, pad_width)), 'constant')
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------------------------------------------------------
# Основная функция
# ----------------------------------------------------------------------
def test_energy():
    categories = ['comp.graphics', 'sci.space']
    newsgroups = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    texts = newsgroups.data[:300]
    labels = newsgroups.target[:300]

    dataset = TextDataset(texts, labels, vocab_size=5000, max_len=64)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Создаём модель с теми же параметрами, что и при обучении:
    # max_len=64, lightweight_ffn=True, ffn_rank=64
    model = CustomTransformer(
        vocab_size=5000,
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        num_classes=2,
        max_len=64,                     # Важно: 64, а не 128
        attention_type='ray',
        num_rays=8,
        lightweight_ffn=True,
        ffn_rank=64,
        sparse_rays=False,
    ).cuda()

    # Загружаем веса
    try:
        state_dict = torch.load('ray_transformer_light.pth')
        # Если размер pos_encoding не совпадает, удалим его (его всё равно перезапишут)
        state_dict.pop('pos_encoding', None)
        model.load_state_dict(state_dict, strict=False)
        print("Loaded ray_transformer_light.pth")
    except FileNotFoundError:
        try:
            state_dict = torch.load('ray_transformer.pth')
            state_dict.pop('pos_encoding', None)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded ray_transformer.pth")
        except FileNotFoundError:
            print("Warning: No trained model found, using random weights")

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
                    x = x.cuda()
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
            'avg_blocks': avg_blocks,
        })
        print(f"Threshold: {thresh}, Total Energy: {report['energy_wh']:.4f} Wh, "
              f"Avg per sample: {avg_energy:.6f} Wh, Avg blocks: {avg_blocks:.2f}")

    print("\n=== Comparison ===")
    if results and results[0]['avg_energy_wh'] > 0:
        base_energy = results[0]['avg_energy_wh']
        for r in results[1:]:
            saving = (1 - r['avg_energy_wh'] / base_energy) * 100
            print(f"Threshold {r['threshold']}: energy saving {saving:.2f}%, "
                  f"average blocks {r['avg_blocks']:.2f} (of {model.num_layers})")
    else:
        print("Base energy is zero, cannot compute savings.")

if __name__ == "__main__":
    test_energy()