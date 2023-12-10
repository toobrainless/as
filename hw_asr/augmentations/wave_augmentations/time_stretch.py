import librosa
import torch
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, rate=1.5):
        self.rate = rate

    def __call__(self, data: Tensor):
        return torch.from_numpy(
            librosa.effects.time_stretch(data.numpy(), rate=self.rate)
        )
