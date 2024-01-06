import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        out = self.extractor(x)
        out = self.squeeze(out)
        out = self.head(out)

        return out
