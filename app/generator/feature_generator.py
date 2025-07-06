import torch.nn as nn


class DummyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(224 * 224 * 3, num_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))
