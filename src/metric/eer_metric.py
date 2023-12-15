import torch

from .base_metric import BaseMetric
from .utils import compute_eer


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, epoch_based=True, **kwargs)
        self.logits = []
        self.targets = []

    def __call__(self, logits, target, **kwargs):
        self.logits.append(logits)
        self.targets.append(target)

        return None

    def calculate(self):
        logits = torch.cat(self.logits)
        targets = torch.cat(self.targets)

        true_scores = logits.detach().cpu().numpy()[..., 1]
        target = targets.detach().cpu().numpy()

        ans = compute_eer(true_scores[target == 1], true_scores[target == 0])[0]
        self.logits = []
        self.targets = []

        return ans
