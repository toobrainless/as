import json
import logging
import os
import sys
from pathlib import Path
from string import ascii_lowercase

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.trainer import Trainer
from src.utils import MetricTracker
from src.utils.object_loading import get_dataloaders
from src.utils.util import ROOT_PATH


@hydra.main(version_base=None, config_path="hw_asr/config", config_name="test")
def main(test_cfg: DictConfig):
    print(f"{get_original_cwd()=}")
    OmegaConf.resolve(test_cfg)
    print(OmegaConf.to_yaml(test_cfg))

    model_ckpt_path = Path(to_absolute_path(ROOT_PATH / test_cfg["checkpoint_path"]))
    print(f"{model_ckpt_path=}")
    model_working_dir = model_ckpt_path.parent
    model_cfg = OmegaConf.load(model_working_dir / ".hydra/config.yaml")
    OmegaConf.resolve(model_cfg)
    print(OmegaConf.to_yaml(model_cfg))

    logging.basicConfig(filename="test.log")
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    alphabet = list(ascii_lowercase + " ")
    text_encoder = instantiate(model_cfg.text_encoder, alphabet=alphabet)

    # setup data_loader instances
    dataloaders = get_dataloaders(test_cfg, text_encoder)

    # build model architecture
    model = instantiate(model_cfg["arch"], n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(test_cfg["checkpoint_path"]))
    checkpoint = torch.load(model_ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    if test_cfg["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    metrics = [
        instantiate(metric, text_encoder=text_encoder) for metric in test_cfg["metrics"]
    ]
    tracker = MetricTracker(*[m for m_list in metrics for m in m_list.get_metrics()])
    with torch.no_grad():
        for name, dataloader in dataloaders.items():
            print(name)
            for batch_num, batch in enumerate(tqdm(dataloader)):
                batch = Trainer.move_batch_to_device(batch, device)
                output = model(**batch)

                if type(output) is dict:
                    batch.update(output)
                else:
                    batch["logits"] = output
                batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
                batch["log_probs_length"] = model.transform_input_lengths(
                    batch["spectrogram_length"]
                )
                batch["probs"] = batch["log_probs"].exp().cpu()
                batch["argmax"] = batch["probs"].argmax(-1)
                for met in metrics:
                    tracker.update(met.name, met(**batch))

            print(f"{name} -- done")
            print("Results:")
            print(tracker.result())

            with open(f"{name}_results.json", "w") as f:
                json.dump(tracker.result(), f, indent=2)
            tracker.reset()


if __name__ == "__main__":
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    logger.info("3.1415926")
    sys.argv.append("hydra.job.chdir=True")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
