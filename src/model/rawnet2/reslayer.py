from torch import nn

from .utils import FMS


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels, out_channels, 3, stride=1, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, 3, stride=1, padding="same"),
        )

        self.proj = nn.Conv1d(in_channels, out_channels, 1, padding="same")

        self.head = nn.Sequential(
            nn.MaxPool1d(3),
            FMS(out_channels),
        )

    def forward(self, x):
        return self.head(self.proj(x) + self.body(x))


class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.blocks = nn.Sequential(
            *(
                [ResBlock(in_channels, out_channels)]
                + [ResBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
            )
        )

    def forward(self, x):
        return self.blocks(x)
