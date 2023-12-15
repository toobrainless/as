import logging
import os
import sys
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def number_of_weights(nn):
    return sum(p.numel() for p in nn.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path="src/config", config_name="train")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logger = logging.getLogger("train")
    print(f"{cfg=}")
    if cfg["resume"] is not None:
        old_output = (Path(get_original_cwd()) / cfg["resume"]).parent
        old_overrides = OmegaConf.load(old_output / ".hydra/overrides.yaml")
        hydra_config = OmegaConf.load(old_output / ".hydra/hydra.yaml")
        current_overrides = HydraConfig.get().overrides.task
        overrides = old_overrides + current_overrides
        print(f"{overrides=}")
        cfg = compose(hydra_config.hydra.job.config_name, overrides=overrides)

    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    dataloaders = get_dataloaders(cfg)

    model = instantiate(cfg["arch"])
    print(f"{number_of_weights(model)=}")
    logger.info(model)

    device, device_ids = prepare_device(cfg["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    loss_module = instantiate(cfg["loss"]).to(device)

    metrics = {
        metric_type: [instantiate(metric) for metric in metrics_list]
        for metric_type, metrics_list in cfg["metrics"].items()
    }

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(cfg["optimizer"], params=trainable_params)
    if cfg.get("lr_scheduler", None) is not None:
        lr_scheduler = instantiate(cfg["lr_scheduler"], optimizer=optimizer)
    else:
        lr_scheduler = None

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer,
        config=cfg,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=cfg["trainer"].get("len_epoch", None),
        keyboard_interrupt_save=cfg.get("keyboard_interrupt_save", False),
    )

    trainer.train()


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=True")
    print("Start training...")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
