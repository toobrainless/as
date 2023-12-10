import torch
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class GaussianNoise(AugmentationBase):
    def __init__(self, scale=0.01):
        self.noise = torch.distributions.Normal(loc=0, scale=scale)

    def __call__(self, data: Tensor):
        x = data + self.noise.sample(data.shape)

        return x
