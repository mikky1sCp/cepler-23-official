# src/cepler/utils/datasets.py
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class TextDataset(Dataset):
    """Датасет для классификации текстов на основе Bag-of-Words."""
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