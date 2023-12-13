import unittest

from hydra import compose, initialize
from hydra.utils import instantiate
from tqdm import tqdm

from src.collate_fn.collate import collate_fn
from src.datasets import LibrispeechDataset
from src.utils.object_loading import get_dataloaders


class TestDataloader(unittest.TestCase):
    def test_collate_fn(self):
        with initialize(version_base=None, config_path="."):
            cfg = compose(config_name="config")
            alphabet = list(" abcdefghijklmnopqrstuvwxyz")
            ds = LibrispeechDataset(
                "dev-clean",
                text_encoder=instantiate(cfg["text_encoder"], alphabet=alphabet),
                cfg=cfg,
            )

            batch_size = 3
            batch = collate_fn([ds[i] for i in range(batch_size)])

            self.assertIn("spectrogram", batch)  # torch.tensor
            batch_size_dim, feature_length_dim, time_dim = batch["spectrogram"].shape
            self.assertEqual(batch_size_dim, batch_size)
            self.assertEqual(feature_length_dim, 128)

            self.assertIn("text_encoded", batch)  # [int] torch.tensor
            # joined and padded indexes representation of transcriptions
            batch_size_dim, text_length_dim = batch["text_encoded"].shape
            self.assertEqual(batch_size_dim, batch_size)

            self.assertIn("text_encoded_length", batch)  # [int] torch.tensor
            # contains lengths of each text entry
            self.assertEqual(len(batch["text_encoded_length"].shape), 1)
            batch_size_dim = batch["text_encoded_length"].shape[0]
            self.assertEqual(batch_size_dim, batch_size)

            self.assertIn("text", batch)  # List[str]
            # simple list of initial normalized texts
            batch_size_dim = len(batch["text"])
            self.assertEqual(batch_size_dim, batch_size)

    def test_dataloaders(self):
        _TOTAL_ITERATIONS = 10
        with initialize(version_base=None, config_path="."):
            cfg = compose(config_name="config")
            alphabet = list(" abcdefghijklmnopqrstuvwxyz")
            dataloaders = get_dataloaders(
                cfg, instantiate(cfg["text_encoder"], alphabet=alphabet)
            )
            for part in ["train", "val"]:
                dl = dataloaders[part]
                for i, batch in tqdm(
                    enumerate(iter(dl)),
                    total=_TOTAL_ITERATIONS,
                    desc=f"Iterating over {part}",
                ):
                    if i >= _TOTAL_ITERATIONS:
                        break
