import math
from pathlib import Path
from typing import List

import torch
import torchaudio
from torch.utils.data import Dataset


class ASVspoofDataset(Dataset):
    def __init__(self, protocol_path, data_path, limit=None, output_samples=64000):
        protocol_path = Path(protocol_path)
        data_path = Path(data_path)

        self.index = []
        self.output_samples = output_samples
        with open(protocol_path, "r") as f:
            for idx, line in enumerate(f.readlines()):
                if limit is not None and idx >= limit:
                    break
                line = line.strip().split()
                self.index.append(
                    {
                        "target": int(line[-1] == "bonafide"),
                        "audio_path": data_path / f"{line[1]}.flac",
                    }
                )

    def __getitem__(self, index):
        item = self.index[index]

        audio = torchaudio.load(item["audio_path"])[0]
        repeats = math.ceil(self.output_samples / audio.shape[1])
        audio = audio.repeat((1, repeats))[:, : self.output_samples]

        return item | {"audio": audio}

    def __len__(self):
        return len(self.index)


class Collator:
    def __call__(self, dataset_items: List[dict]):
        result_batch = {}
        result_batch["audio_path"] = []
        result_batch["audio"] = []
        result_batch["target"] = []

        for item in dataset_items:
            result_batch["audio_path"].append(item["audio_path"])
            result_batch["audio"].append(item["audio"])
            result_batch["target"].append(item["target"])

        result_batch["audio"] = torch.concat(result_batch["audio"])
        result_batch["target"] = torch.tensor(result_batch["target"])

        return result_batch
