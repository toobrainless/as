import logging
import os
import sys
import warnings
from pathlib import Path
from string import ascii_lowercase

import hydra
import numpy as np
import torch
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from hw_asr.trainer import Trainer
from hw_asr.utils import prepare_device
from hw_asr.utils.object_loading import get_dataloaders

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(version_base=None, config_path="hw_asr/config", config_name="train")
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

    alphabet = list(ascii_lowercase + " ")
    text_encoder = instantiate(cfg.text_encoder, alphabet=alphabet)

    # setup data_loader instances
    dataloaders = get_dataloaders(cfg, text_encoder)

    # build model architecture, then print to console
    model = instantiate(cfg["arch"], n_class=len(text_encoder))
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = instantiate(cfg["loss"], blank=len(alphabet)).to(device)
    metrics = {
        metric_type: [
            instantiate(metric, text_encoder=text_encoder) for metric in metrics_list
        ]
        for metric_type, metrics_list in cfg["metrics"].items()
    }

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
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
        text_encoder=text_encoder,
        config=cfg,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=cfg["trainer"].get("len_epoch", None),
        keyboard_interrupt_save=cfg["keyboard_interrupt_save"],
    )

    if cfg["mode"] == "train":
        trainer.train()
    elif cfg["mode"] == "profile":
        from torch.profiler import ProfilerActivity, profile, record_function

        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                trainer._train_epoch(0)
        with open("profiler_results.txt", "w") as f:
            print(
                prof.key_averages().table(sort_by="cpu_time_total", row_limit=10),
                file=f,
            )
    else:
        assert False, "wrong mode"


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=True")
    print("start training")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
