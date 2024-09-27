import torch.nn as nn


class DistanceMLP(nn.Module):
    def __init__(self, in_channels=512, out_channels=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, out_channels),
        )

    def forward(self, x):
        return self.layer(x)


class MemoryMLP(nn.Module):
    def __init__(self, in_channels=512, out_channels=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, out_channels),
        )

    def forward(self, x):
        return self.layer(x)
