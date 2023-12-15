class BaseMetric:
    def __init__(self, name: str = None, epoch_based: bool = False, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.epoch_based = epoch_based

    def get_metrics(self):
        return [self.name]

    def __call__(self, **batch):
        raise NotImplementedError()
