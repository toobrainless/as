class BaseMetric:
    def __init__(self, name=None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__

    def get_metrics(self):
        return [self.name]

    def __call__(self, **batch):
        raise NotImplementedError()
