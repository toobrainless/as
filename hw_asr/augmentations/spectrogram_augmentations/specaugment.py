import torch

from ..base import AugmentationBase


class SpecAug(AugmentationBase):
    def __init__(self, freq_masks, time_masks, freq_width, time_width):
        super().__init__()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width

    def __call__(self, spectrogram):
        batch_size, frequency, time = spectrogram.shape
        for i in range(batch_size):
            for j in range(self.freq_masks):
                f = torch.randint(low=0, high=self.freq_width, size=(1,)).item()
                f_0 = torch.randint(low=0, high=frequency - f, size=(1,)).item()
                spectrogram[:, f_0 : f_0 + f, :] = 0
            for j in range(self.time_masks):
                t = torch.randint(
                    low=0, high=int(spectrogram.shape[1] * self.time_width), size=(1,)
                ).item()
                t_0 = torch.randint(low=0, high=time - t, size=(1,)).item()
                spectrogram[:, :, t_0 : t_0 + t] = 0
        return spectrogram
