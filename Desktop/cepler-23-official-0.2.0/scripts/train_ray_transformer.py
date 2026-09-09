# scripts/train_ray_transformer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from cepler.models.transformer import CustomTransformer
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import argparse

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

def train(attention_type='ray', lightweight_ffn=False, aux_weight=0.7):
    categories = ['comp.graphics', 'sci.space']
    newsgroups = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    texts = newsgroups.data[:2000]
    labels = newsgroups.target[:2000]

    dataset = TextDataset(texts, labels, vocab_size=5000, max_len=64)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    model = CustomTransformer(
        vocab_size=5000, d_model=256, num_heads=8, d_ff=512,
        num_layers=4, num_classes=2, max_len=64,
        attention_type=attention_type, num_rays=8,
        lightweight_ffn=lightweight_ffn,
        ffn_rank=64 if lightweight_ffn else None
    ).cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    
    model.train()
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with autocast('cuda'):
                all_logits, _ = model(x, return_all_logits=True)
                loss_main = criterion(all_logits[-1], y)
                loss_aux = 0
                for logits in all_logits[:-1]:
                    loss_aux += criterion(logits, y)
                loss_aux = loss_aux / len(all_logits[:-1])
                loss = loss_main + aux_weight * loss_aux
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"{attention_type} | Epoch {epoch+1}/5, Loss: {total_loss/len(loader):.4f}")

    suffix = "_light" if lightweight_ffn else ""
    torch.save(model.state_dict(), f'{attention_type}_transformer{suffix}.pth')
    print(f"Model saved as {attention_type}_transformer{suffix}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lightweight_ffn', action='store_true', default=False)
    parser.add_argument('--aux_weight', type=float, default=0.7)
    args = parser.parse_args()

    print("Training Ray with lightweight_ffn=", args.lightweight_ffn)
    train('ray', lightweight_ffn=args.lightweight_ffn, aux_weight=args.aux_weight)
    print("\nTraining Multihead with lightweight_ffn=", args.lightweight_ffn)
    train('multihead', lightweight_ffn=args.lightweight_ffn, aux_weight=args.aux_weight)