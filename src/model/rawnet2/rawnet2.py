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
        reslayer1_out_channels=20,
        reslayer2_out_channels=128,
        sinc_abs_after=True,
        sinc_min_low_hz=0,
        sinc_min_band_hz=0,
        sinc_requires_grad=False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            SincConv_fast(
                out_channels=128,
                kernel_size=129,
                min_low_hz=sinc_min_low_hz,
                min_band_hz=sinc_min_band_hz,
                requires_grad=sinc_requires_grad,
            ),
            SwitchABS(sinc_abs_after),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            ResLayer(
                in_channels=128, out_channels=reslayer1_out_channels, num_blocks=2
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
            hidden_size=reslayer2_out_channels * 2,
            input_size=reslayer2_out_channels,
        )
        self.fc = nn.Linear(reslayer2_out_channels * 2, 2)

    def forward(self, audio, **kwargs):
        x = audio.unsqueeze(1)
        x = self.net(x)
        x = self.gru(x.transpose(-1, -2))[0][:, -1, :]
        x = self.fc(x)

        return x
