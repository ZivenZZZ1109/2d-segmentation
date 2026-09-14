import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from seg2d.toy import create_toy_data

try:
    import torch
except ImportError:
    torch = None


class ToyDataTests(unittest.TestCase):
    def test_generation_is_reproducible_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            first = create_toy_data(Path(directory) / "first", counts=(2, 1, 1))
            second = create_toy_data(Path(directory) / "second", counts=(2, 1, 1))
            for path in first.rglob("*.png"):
                self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())
            with self.assertRaises(FileExistsError):
                create_toy_data(first)


@unittest.skipIf(torch is None, "Install the model extra to run training tests.")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        from seg2d.training import TrainConfig

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = create_toy_data(self.root / "data", size=32, counts=(3, 2, 2))
        self.config = TrainConfig(size=32, base_channels=2, epochs=2, batch_size=2)

    def run_training(self, output="training"):
        from seg2d.training import train

        with contextlib.redirect_stdout(io.StringIO()):
            return train(self.data, self.root / output, self.config)

    def test_training_checkpoint_metrics_and_reproducibility(self):
        from seg2d.inference import load_model, predict

        report = self.run_training()
        self.assertEqual(report["status"], "completed")
        self.assertIn(report["best_epoch"], (1, 2))
        self.assertEqual(report["validation"]["images"], 2)
        output = self.root / "training"
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "completed")
        history = np.genfromtxt(output / "history.csv", delimiter=",", names=True)
        self.assertAlmostEqual(report["validation"]["loss"], float(history["val_loss"].min()))
        first = torch.load(output / "best.pt", weights_only=True)
        self.assertEqual(first["epoch"], report["best_epoch"])
        self.assertEqual(first["config"]["size"], 32)
        self.run_training("repeat")
        second = torch.load(self.root / "repeat" / "best.pt", weights_only=True)
        for name in first["state_dict"]:
            self.assertTrue(torch.equal(first["state_dict"][name], second["state_dict"][name]), name)
        model = load_model(output / "best.pt", 3, 2)
        self.assertEqual(predict(model, np.zeros((32, 32, 3), dtype=np.uint8), 32).shape, (32, 32))
        original = (output / "best.pt").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_training()
        self.assertEqual((output / "best.pt").read_bytes(), original)

    def test_pairing_rejects_missing_and_extra_masks(self):
        from seg2d.data import PairedDataset

        (self.data / "train" / "masks" / "000.png").rename(self.data / "train" / "masks" / "extra.png")
        with self.assertRaisesRegex(ValueError, "pairing"):
            PairedDataset(self.data / "train", 3, 32)

    def test_duplicate_stems_rejected(self):
        from seg2d.data import PairedDataset

        shutil.copyfile(self.data / "train" / "images" / "000.png",
                        self.data / "train" / "images" / "000.jpg")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            PairedDataset(self.data / "train", 3, 32)

    def test_out_of_range_and_misaligned_masks_rejected(self):
        from seg2d.data import PairedDataset

        mask = self.data / "train" / "masks" / "000.png"
        Image.fromarray(np.full((32, 32), 256, dtype=np.uint16)).save(mask)
        with self.assertRaisesRegex(ValueError, "class indices"):
            PairedDataset(self.data / "train", 3, 32)
        Image.fromarray(np.zeros((31, 32), dtype=np.uint8)).save(mask)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            PairedDataset(self.data / "train", 3, 32)

    def test_split_overlap_rejected_before_training(self):
        shutil.copyfile(self.data / "train" / "images" / "000.png",
                        self.data / "val" / "images" / "001.png")
        with self.assertRaisesRegex(ValueError, "across dataset splits"):
            self.run_training()
        self.assertFalse((self.root / "training").exists())

    def test_validation_is_deterministic_and_transforms_aligned(self):
        from seg2d.data import PairedDataset

        mask = np.asarray(Image.open(self.data / "train" / "masks" / "000.png"))
        encoded = np.repeat((mask * 80)[:, :, None], 3, axis=2)
        Image.fromarray(encoded).save(self.data / "train" / "images" / "000.png")
        dataset = PairedDataset(self.data / "train", 3, 32, augment=True, seed=3)
        for _ in range(8):
            image, target = dataset[0]
            torch.testing.assert_close(image[0] * 255, target.float() * 80)
        validation = PairedDataset(self.data / "val", 3, 32)
        first, second = validation[0], validation[0]
        self.assertFalse(validation.augment)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))

    def test_bad_checkpoint_does_not_overwrite_healthy_file(self):
        from seg2d.model import UNet
        from seg2d.training import save_checkpoint

        model, path = UNet(n_classes=3, base_channels=2), self.root / "best.pt"
        save_checkpoint(model, path, self.config, 1, 0.5)
        original = path.read_bytes()
        with self.assertRaises(FloatingPointError):
            save_checkpoint(model, path, self.config, 2, float("nan"))
        model.inc.double_conv[1].running_mean[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            save_checkpoint(model, path, self.config, 2, 0.4)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_failed_atomic_write_preserves_checkpoint(self):
        from seg2d.model import UNet
        from seg2d.training import save_checkpoint

        model, path = UNet(n_classes=3, base_channels=2), self.root / "best.pt"
        save_checkpoint(model, path, self.config, 1, 0.5)
        original = path.read_bytes()
        with patch("seg2d.training.torch.save", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                save_checkpoint(model, path, self.config, 2, 0.4)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_training_failure_is_explicit_and_never_marked_complete(self):
        with patch("seg2d.training.evaluate", side_effect=FloatingPointError("non-finite logits")):
            with self.assertRaises(FloatingPointError):
                self.run_training()
        output = self.root / "training"
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")
        self.assertFalse((output / "report.json").exists())
        self.assertFalse((output / "best.pt").exists())

    def test_test_directory_does_not_participate_in_training(self):
        for path in (self.data / "test" / "masks").glob("*.png"):
            path.write_bytes(b"invalid test data, never read by the trainer")
        self.assertEqual(self.run_training()["status"], "completed")

    def test_failure_after_healthy_epoch_preserves_best(self):
        from seg2d.training import evaluate

        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise FloatingPointError("non-finite validation")
            return evaluate(*args, **kwargs)

        with patch("seg2d.training.evaluate", side_effect=fail_second):
            with self.assertRaises(FloatingPointError):
                self.run_training()
        output = self.root / "training"
        checkpoint = torch.load(output / "best.pt", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in checkpoint["state_dict"].values()))
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")

    def test_equal_loss_counts_towards_early_stopping(self):
        from dataclasses import replace
        from seg2d.training import evaluate

        self.config = replace(self.config, epochs=5, patience=1)

        def fixed_loss(*args, **kwargs):
            metrics = evaluate(*args, **kwargs)
            metrics["loss"] = 1.0
            return metrics

        with patch("seg2d.training.evaluate", side_effect=fixed_loss):
            report = self.run_training()
        self.assertEqual(report["best_epoch"], 1)
        self.assertEqual(report["epochs_run"], 2)

    def test_cli_training(self):
        from seg2d.cli import main

        with contextlib.redirect_stdout(io.StringIO()):
            main(["train", str(self.data), str(self.root / "cli"), "--classes", "3",
                  "--size", "32", "--base-channels", "2", "--epochs", "1", "--no-augment"])
        config = json.loads((self.root / "cli" / "config.json").read_text())
        self.assertFalse(config["augment"])
        self.assertTrue((self.root / "cli" / "best.pt").is_file())

    def test_invalid_configuration_rejected(self):
        from dataclasses import replace

        for change in ({"epochs": 0}, {"size": 16}, {"learning_rate": float("nan")},
                       {"num_classes": 257}, {"seed": -1}, {"device": "automatic"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config, **change).validate()
