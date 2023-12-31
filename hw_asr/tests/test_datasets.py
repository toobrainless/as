import unittest

import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from hw_asr.datasets import (CustomAudioDataset, CustomDirAudioDataset,
                             LibrispeechDataset)
from hw_asr.utils import ROOT_PATH


class TestDataset(unittest.TestCase):
    def test_librispeech(self):
        with initialize(version_base=None, config_path="."):
            cfg = compose(config_name="config")
            alphabet = list(" abcdefghijklmnopqrstuvwxyz")
            ds = LibrispeechDataset(
                "dev-clean",
                text_encoder=instantiate(cfg["text_encoder"], alphabet=alphabet),
                cfg=cfg,
                max_text_length=140,
                max_audio_length=13,
                limit=10,
            )
            self._assert_training_example_is_good(ds[0])

    def test_custom_dir_dataset(self):
        with initialize(version_base=None, config_path="."):
            cfg = compose(config_name="config")
            audio_dir = str(ROOT_PATH / "test_data" / "audio")
            transc_dir = str(ROOT_PATH / "test_data" / "transcriptions")
            alphabet = list(" abcdefghijklmnopqrstuvwxyz")
            ds = CustomDirAudioDataset(
                audio_dir,
                transc_dir,
                text_encoder=instantiate(cfg["text_encoder"], alphabet=alphabet),
                cfg=cfg,
                limit=10,
                max_audio_length=8,
                max_text_length=130,
            )
            self._assert_training_example_is_good(ds[0])

    def test_custom_dataset(self):
        with initialize(version_base=None, config_path="."):
            cfg = compose(config_name="config")
            audio_path = ROOT_PATH / "test_data" / "audio"
            transc_path = ROOT_PATH / "test_data" / "transcriptions"
            with (transc_path / "84-121550-0000.txt").open() as f:
                transcription = f.read().strip()
            data = [
                {
                    "path": str(audio_path / "84-121550-0001.flac"),
                },
                {
                    "path": str(audio_path / "84-121550-0000.flac"),
                    "text": transcription,
                },
            ]
            alphabet = list(" abcdefghijklmnopqrstuvwxyz")
            ds = CustomAudioDataset(
                data=data,
                text_encoder=instantiate(cfg["text_encoder"], alphabet=alphabet),
                cfg=cfg,
            )
            self._assert_training_example_is_good(ds[0], contains_text=False)
            self._assert_training_example_is_good(ds[1])

    def _assert_training_example_is_good(
        self, training_example: dict, contains_text=True
    ):
        for field, expected_type in [
            ("audio", torch.Tensor),
            ("spectrogram", torch.Tensor),
            ("duration", float),
            ("audio_path", str),
            ("text", str),
            ("text_encoded", torch.Tensor),
        ]:
            self.assertIn(
                field, training_example, f"Error during checking field {field}"
            )
            self.assertIsInstance(
                training_example[field],
                expected_type,
                f"Error during checking field {field}",
            )

        # check waveform dimensions
        (
            batch_dim,
            audio_dim,
        ) = training_example["audio"].size()
        self.assertEqual(batch_dim, 1)
        self.assertGreater(audio_dim, 1)

        # check spectrogram dimensions
        batch_dim, freq_dim, time_dim = training_example["spectrogram"].size()
        self.assertEqual(batch_dim, 1)
        self.assertEqual(freq_dim, 128)
        self.assertGreater(time_dim, 1)

        # check text tensor dimensions
        (
            batch_dim,
            length_dim,
        ) = training_example["text_encoded"].size()
        self.assertEqual(batch_dim, 1)
        if contains_text:
            self.assertGreater(length_dim, 1)
        else:
            self.assertEqual(length_dim, 0)
            self.assertEqual(training_example["text"], "")
