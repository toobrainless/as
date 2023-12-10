from collections.abc import Callable
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from .sequential import SequentialAugmentation


def from_configs(cfg: DictConfig):
    wave_augs = []
    if "augmentations" in cfg and "wave" in cfg["augmentations"]:
        for aug in cfg["augmentations"]["wave"]:
            wave_augs.append(instantiate(aug))

    spec_augs = []
    if "augmentations" in cfg and "spectrogram" in cfg["augmentations"]:
        for aug in cfg["augmentations"]["spectrogram"]:
            spec_augs.append(instantiate(aug))

    return _to_function(wave_augs), _to_function(spec_augs)


def _to_function(augs_list: List[Callable]):
    if len(augs_list) == 0:
        return None
    elif len(augs_list) == 1:
        return augs_list[0]
    else:
        return SequentialAugmentation(augs_list)
