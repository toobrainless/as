import logging
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


# TODO write callable class
def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    result_batch["text_encoded_length"] = []
    result_batch["audio_length"] = []
    result_batch["spectrogram_length"] = []
    result_batch["audio_path"] = []
    result_batch["text"] = []
    result_batch["duration"] = []
    result_batch["spectrogram"] = []
    result_batch["audio"] = []
    result_batch["text_encoded"] = []

    for item in dataset_items:
        result_batch["audio_length"].append(item["audio"].shape[1])
        result_batch["spectrogram_length"].append(item["spectrogram"].shape[2])
        result_batch["text_encoded_length"].append(item["text_encoded"].shape[1])
        result_batch["audio_path"].append(item["audio_path"])
        result_batch["text"].append(item["text"])
        result_batch["duration"].append(item["duration"])
        result_batch["spectrogram"].append(item["spectrogram"][0].permute(1, 0))
        result_batch["audio"].append(item["audio"][0])
        result_batch["text_encoded"].append(item["text_encoded"][0])

    result_batch["audio_length"] = torch.tensor(result_batch["audio_length"])
    result_batch["spectrogram_length"] = torch.tensor(
        result_batch["spectrogram_length"]
    )
    result_batch["text_encoded_length"] = torch.tensor(
        result_batch["text_encoded_length"]
    )

    result_batch["spectrogram"] = pad_sequence(
        result_batch["spectrogram"], batch_first=True
    ).permute(0, 2, 1)
    result_batch["audio"] = pad_sequence(result_batch["audio"], batch_first=True)
    result_batch["text_encoded"] = pad_sequence(
        result_batch["text_encoded"], batch_first=True
    )

    return result_batch
