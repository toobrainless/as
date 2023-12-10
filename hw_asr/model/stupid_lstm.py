from torch import nn
from torch.nn import LSTM, Sequential

from .base_model import BaseModel


class StupidLSTM(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.n_class = n_class
        self.net_1 = LSTM(
            input_size=n_feats,
            hidden_size=fc_hidden,
            num_layers=4,
            batch_first=True,
            # proj_size=n_class // 2,
            bidirectional=True,
            dropout=0.2,
        )
        self.layer_norm = nn.LayerNorm(2 * fc_hidden)
        self.net_2 = LSTM(
            input_size=2 * fc_hidden,
            hidden_size=fc_hidden,
            num_layers=4,
            batch_first=True,
            proj_size=n_class // 2,
            bidirectional=True,
            dropout=0.2,
        )
        # self.head = nn.Linear(n_feats * 4, n_class)

    def forward(self, spectrogram, **batch):
        # print(f'{spectrogram.shape=}')
        input = spectrogram.permute((0, 2, 1))
        # print(f'{input.shape=}')
        output = self.net_1(input)[0]
        output = self.layer_norm(output)
        output = self.net_2(output)[0]
        # print(f'{output.shape=}')
        return {"logits": output}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
