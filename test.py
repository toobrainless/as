import math
from pathlib import Path

import click
import pandas as pd
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn import functional as F


def load_audio(path):
    OUTPUT_SAMPLES = 64000
    audio = torchaudio.load(path)[0]
    repeats = math.ceil(OUTPUT_SAMPLES / audio.shape[1])
    audio = audio.repeat((1, repeats))[:, :OUTPUT_SAMPLES]

    return audio
    return audio


@click.command()
@click.option(
    "--model_path",
    "-mp",
    default="small",
    help="Path to folder with model.",
    type=str,
)
@click.option(
    "--test_audio_path",
    "-tap",
    default="test_audio",
    help="Path to folder with test audio.",
    type=str,
)
def main(model_path, test_audio_path):
    model_path = Path(model_path)
    audio_batch = []
    audio_stems = []

    test_audio_path = Path(test_audio_path)
    for path in sorted(test_audio_path.iterdir()):
        audio_stems.append(path.stem)
        audio_batch.append(load_audio(path))

    audio_batch = torch.concatenate(audio_batch)

    with initialize(version_base=None, config_path=model_path):
        cfg = compose(config_name="config")
        OmegaConf.resolve(cfg)

    model = instantiate(cfg["arch"])
    model.load_state_dict(torch.load(model_path / "checkpoint.pth")["state_dict"])

    logits = model(audio=audio_batch)
    probabilities = F.softmax(logits, dim=1).tolist()

    results = pd.DataFrame(columns=["stem", "spoof_prob", "bonified_prob"])

    for idx, (stem, probability) in enumerate(zip(audio_stems, probabilities)):
        results.loc[idx] = [stem, *list(map(lambda x: f"{x:.7f}", probability))]

    results.to_csv(f"{model_path.stem}_results.csv")
    print(results)


if __name__ == "__main__":
    main()
