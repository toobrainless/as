import torch
from torch import nn


class Logger(nn.Module):
    def __init__(self, text=""):
        super().__init__()
        self.text = text

    def forward(self, x):
        print(f"{self.text}: {x.shape=}")
        return x


class FMS(nn.Module):
    def __init__(self, filters_num):
        super().__init__()
        self.fc = nn.Linear(filters_num, filters_num)

    def forward(self, x):
        return x * torch.sigmoid(self.fc(x.mean(-1))).unsqueeze(-1)


class SwitchABS(nn.Module):
    def __init__(self, is_on=True):
        super().__init__()
        self.is_on = is_on

    def forward(self, x):
        if self.is_on:
            return torch.abs(x)
        else:
            x
