import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class FeatureDataset(Dataset):
    def __init__(self, feature_file, label_file):
        self.features = np.load(feature_file)
        self.labels = np.load(label_file)
        assert len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Optionally convert to torch.Tensor here for speed
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

# Example usage:
train_dataset = FeatureDataset('feature_train_1.npy', 'label_train_1.npy')

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

for features, labels in train_loader:
    # features: [64, feature_dim], labels: [64]
    pass