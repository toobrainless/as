from functools import partial
from typing import List

import numpy as np
import torch
from torch import Tensor

from hw_asr.text_encoder import BaseTextEncoder

from .base_metric import BaseMetric
from .utils import calc_cer, calc_wer


class WERCERBaseMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = ["WER", "CER"]

    def get_metrics(self):
        return [f"{self.name}_{metric}" for metric in self.metrics]

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        cers = []
        for log_probs, length, target_text in zip(log_probs, log_probs_length, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            pred_text = self._decode(log_probs.cpu().numpy(), length)
            wers.append(calc_wer(target_text, pred_text))
            cers.append(calc_cer(target_text, pred_text))
        return {"WER": sum(wers) / len(wers), "CER": sum(cers) / len(cers)}


class ArgmaxWERCERMetric(WERCERBaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._decode = text_encoder.ctc_argmax


class CustomBeamSearchWERCERMetric(WERCERBaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._decode = partial(text_encoder.custom_ctc_beam_search, beam_size=beam_size)


class BeamSearchWERCERMetric(WERCERBaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._decode = partial(text_encoder.custom_ctc_beam_search, beam_size=beam_size)


class LMBeamSearchWERCERMetric(WERCERBaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._decode = partial(text_encoder.ptctc_beam_search_lm, beam_size=beam_size)
