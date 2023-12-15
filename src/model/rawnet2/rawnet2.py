import torch
from torch import nn

from .reslayer import ResLayer
from .sinclayer import SincConv_fast
from .utils import SwitchABS


class RawNet2(nn.Module):
    """
    We have hints and some restrictions for parameters,
    so only the parameters needed for experimentation are available.
    """

    def __init__(
        self,
        gru_prenorm=True,
        gru_num_layers=3,
        gru_hidden_size=1024,
        reslayer1_out_channels=20,
        reslayer2_out_channels=128,
        sinc_abs_after=True,
        sinc_min_low_hz=0,
        sinc_min_band_hz=0,
        sinc_requires_grad=False,
        sinc_kernel_size=1024,
    ):
        super().__init__()
        self.net = nn.Sequential(
            SincConv_fast(
                out_channels=reslayer1_out_channels,
                kernel_size=sinc_kernel_size,
                min_low_hz=sinc_min_low_hz,
                min_band_hz=sinc_min_band_hz,
                requires_grad=sinc_requires_grad,
            ),
            SwitchABS(sinc_abs_after),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(reslayer1_out_channels),
            nn.LeakyReLU(),
            ResLayer(
                in_channels=reslayer1_out_channels,
                out_channels=reslayer1_out_channels,
                num_blocks=2,
            ),
            ResLayer(
                in_channels=reslayer1_out_channels,
                out_channels=reslayer2_out_channels,
                num_blocks=4,
            ),
        )

        if gru_prenorm:
            self.net.append(nn.BatchNorm1d(reslayer2_out_channels))
            self.net.append(nn.LeakyReLU())

        self.gru = nn.GRU(
            batch_first=True,
            num_layers=gru_num_layers,
            hidden_size=gru_hidden_size,
            input_size=reslayer2_out_channels,
        )
        self.fc = nn.Linear(gru_hidden_size, gru_hidden_size)
        self.fc_out = nn.Linear(gru_hidden_size, 2)

    def forward(self, audio, **kwargs):
        x = audio.unsqueeze(1)
        x = self.net(x)
        x = self.gru(x.transpose(-1, -2))[0][:, -1, :]
        x = self.fc(x)
        norm = x.norm(p=2, dim=1, keepdim=True) / 10.0
        x = torch.div(x, norm)
        x = self.fc_out(self.fc(x))

        return x
