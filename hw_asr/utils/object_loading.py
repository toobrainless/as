from operator import xor

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader

import hw_asr.augmentations
import hw_asr.datasets
from hw_asr.collate_fn.collate import collate_fn
from hw_asr.text_encoder.base_text_encoder import BaseTextEncoder


def get_dataloaders(cfg: DictConfig, text_encoder: BaseTextEncoder):
    dataloaders = {}
    for split, params in cfg["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == "train":
            wave_augs, spec_augs = hw_asr.augmentations.utils.from_configs(cfg)
            drop_last = True
        else:
            wave_augs, spec_augs = None, None
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(
                instantiate(
                    config=ds,
                    cfg=cfg,
                    text_encoder=text_encoder,
                    wave_augs=wave_augs,
                    spec_augs=spec_augs,
                    _recursive_=False,
                )
            )
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor(
            "batch_size" in params, "batch_sampler" in params
        ), "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        elif "batch_sampler" in params:
            batch_sampler = instantiate(params["batch_sampler"], data_source=dataset)
            bs, shuffle = 1, False
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(
            dataset
        ), f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            drop_last=drop_last,
        )
        dataloaders[split] = dataloader
    return dataloaders
