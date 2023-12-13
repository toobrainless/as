from typing import List

from .base_metric import BaseMetric
from .utils import compute_eer


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits, target, **kwargs):
        true_scores = logits.detach().cpu().numpy()[..., 1]
        target = target.detach().cpu().numpy()

        print(f"{logits.shape=}")
        print(f"{target.shape=}")

        return compute_eer(true_scores[target == 1], true_scores[target == 0])[0]
