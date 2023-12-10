import torch
from torch import nn

from .block import ConformerBlock


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        encoder_dim,
        n_layers,
        num_heads,
        dropout,
        kernel_size,
        n_class,
        subsampling_factor=2,
    ):
        super().__init__()
        self.subsampling = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.fc1 = nn.Linear(embed_dim // subsampling_factor, encoder_dim)

        self.dropout = nn.Dropout(0.1)

        self.blocks = nn.Sequential(
            *[
                ConformerBlock(encoder_dim, dropout, num_heads, kernel_size)
                for _ in range(n_layers)
            ]
        )

        self.fc2 = nn.Linear(encoder_dim, n_class)

    def forward(self, spectrogram, **batch):
        x = spectrogram.transpose(-1, -2)
        x = self.subsampling(x.unsqueeze(1)).squeeze(1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.fc2(x)

        output = {"logits": x}

        return output

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
