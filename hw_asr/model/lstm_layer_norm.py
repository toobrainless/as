import torch
from torch import nn

from .base_model import BaseModel


class LstmLayerNormModel(BaseModel):
    def __init__(
        self,
        n_feats,
        n_class,
        fc_hidden=512,
        num_layers=4,
        dropout=0.0,
        batch_first=False,
        **batch,
    ):
        super().__init__(n_feats, n_class, **batch)
        input_size = n_feats
        output_size = n_class
        hidden_size = fc_hidden
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

        self.cells = nn.ModuleList()
        self.ln = nn.ModuleList()
        for _ in range(num_layers):
            self.cells.append(
                nn.LSTMCell(
                    input_size=input_size if _ == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
            )
            self.ln.append(nn.LayerNorm(hidden_size))

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, spectrogram, init_states=None, **batch):
        x = spectrogram.permute((0, 2, 1))
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        if init_states is None:
            init_states = self._init_states(batch_size)
            init_states = [(h.to(x.device), c.to(x.device)) for h, c in init_states]

        layer_outputs = []
        for i in range(self.num_layers):
            layer_input = x if i == 0 else layer_outputs[-1]
            h_t, c_t = init_states[i]
            outputs = []
            for t in range(seq_len):
                h_t, c_t = self.cells[i](layer_input[t], (h_t, c_t))
                outputs.append(h_t)
            outputs = torch.stack(outputs)
            outputs = self.ln[i](outputs)
            outputs = self.dropout(outputs) if i < self.num_layers - 1 else outputs
            layer_outputs.append(outputs)

        output = self.output_layer(layer_outputs[-1])

        # Transpose the output to (batch, seq, feature) if batch_first is True
        if self.batch_first:
            output = output.transpose(0, 1)

        return {"logits": output}

    def _init_states(self, batch_size):
        states = []
        for _ in range(self.num_layers):
            h_t = torch.zeros(batch_size, self.hidden_size)
            c_t = torch.zeros(batch_size, self.hidden_size)
            states.append((h_t, c_t))
        return states

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
