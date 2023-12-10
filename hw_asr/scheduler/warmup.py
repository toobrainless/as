from torch.optim.lr_scheduler import _LRScheduler


class WarmUpScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1, min_lr=0.0):
        self.min_lr = min_lr
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        return [
            max(
                self.min_lr,
                base_lr
                * self.d_model ** (-0.5)
                * min(
                    step ** (-0.5),
                    step * self.warmup_steps ** (-1.5),
                ),
            )
            for base_lr in self.base_lrs
        ]
