"""Small FP32 trainer with validation-only selection and fail-fast checkpoints."""

import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import PairedDataset, require_disjoint
from .images import require_new_output
from .inference import load_model
from .metrics import ConfusionMatrix
from .model import UNet


@dataclass(frozen=True)
class TrainConfig:
    num_classes: int = 3
    size: int = 64
    base_channels: int = 4
    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 0.003
    patience: int = 8
    seed: int = 7
    device: str = "cpu"
    threads: int = 1
    augment: bool = True

    def validate(self):
        for name in ("base_channels", "epochs", "batch_size", "patience", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.num_classes) is not int or not 2 <= self.num_classes <= 256:
            raise ValueError("num_classes must be in [2, 256].")
        if type(self.size) is not int or self.size < 32:
            raise ValueError("size must be >= 32.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda.")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require_finite_model(model: nn.Module) -> None:
    for name, value in model.state_dict().items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite model tensor: {name}")


def save_checkpoint(model: nn.Module, path: Path, config: TrainConfig,
                    epoch: int, val_loss: float) -> None:
    if not math.isfinite(val_loss):
        raise FloatingPointError("Refusing to save a checkpoint with non-finite validation loss.")
    require_finite_model(model)
    payload = {
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "config": asdict(config), "epoch": epoch, "val_loss": val_loss,
    }
    # Write beside the destination, so replace stays on the same filesystem.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device: str) -> dict:
    model.eval()
    metric = ConfusionMatrix(num_classes)
    total_loss, pixels = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Non-finite evaluation logits.")
        loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite evaluation loss.")
        total_loss += float(loss)
        pixels += labels.numel()
        metric.update(labels.cpu().numpy(), logits.argmax(1).cpu().numpy())
    result = metric.compute()
    result["loss"] = total_loss / pixels
    result["images"] = len(loader.dataset)
    result["resolution"] = "resized dataset grid; not restored original-size masks"
    return result


def train(data_dir: str | Path, output_dir: str | Path, config: TrainConfig) -> dict:
    config.validate()
    root, output = Path(data_dir), require_new_output(output_dir)
    training = PairedDataset(root / "train", config.num_classes, config.size,
                             augment=config.augment, seed=config.seed)
    validation = PairedDataset(root / "val", config.num_classes, config.size)
    require_disjoint(training, validation)
    output.mkdir(parents=True)
    write_json(output / "config.json", asdict(config))
    write_json(output / "split_manifest.json", {"train": training.manifest, "val": validation.manifest})
    write_json(output / "environment.json", {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "device": config.device, "precision": "float32",
        "num_workers": 0,
    })
    previous_threads = torch.get_num_threads()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_deterministic = torch.backends.cudnn.deterministic
    started = time.perf_counter()
    try:
        torch.set_num_threads(config.threads)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        train_loader = DataLoader(training, batch_size=config.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(config.seed), num_workers=0)
        val_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False, num_workers=0)
        model = UNet(n_classes=config.num_classes, base_channels=config.base_channels).to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        best_loss, best_epoch, stale = math.inf, 0, 0
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_mean_iou"])
            writer.writeheader()
            for epoch in range(1, config.epochs + 1):
                model.train()
                total_loss, pixels = 0.0, 0
                for images, labels in train_loader:
                    images, labels = images.to(config.device), labels.to(config.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = nn.functional.cross_entropy(logits, labels)
                    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("Non-finite training loss/logits; stopping this run.")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, error_if_nonfinite=True)
                    optimizer.step()
                    require_finite_model(model)
                    total_loss += float(loss.detach()) * labels.numel()
                    pixels += labels.numel()
                metrics = evaluate(model, val_loader, config.num_classes, config.device)
                writer.writerow({"epoch": epoch, "train_loss": total_loss / pixels,
                                 "val_loss": metrics["loss"], "val_mean_iou": metrics["mean_iou"]})
                handle.flush()
                print(f"Epoch {epoch}/{config.epochs} train={total_loss / pixels:.4f} "
                      f"val={metrics['loss']:.4f} val_mIoU={metrics['mean_iou']:.4f}", flush=True)
                if metrics["loss"] < best_loss:
                    save_checkpoint(model, output / "best.pt", config, epoch, metrics["loss"])
                    best_loss, best_epoch, stale = metrics["loss"], epoch, 0
                else:
                    stale += 1
                    if stale >= config.patience:
                        break
        best_model = load_model(output / "best.pt", config.num_classes, config.base_channels, config.device)
        report = {"status": "completed", "best_epoch": best_epoch, "epochs_run": epoch,
                  "selection": "lowest finite validation cross-entropy loss",
                  "validation": evaluate(best_model, val_loader, config.num_classes, config.device),
                  "elapsed_seconds": time.perf_counter() - started}
        write_json(output / "report.json", report)
        write_json(output / "status.json", {"status": "completed"})
        return report
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__,
                                            "message": str(error)})
        raise
    finally:
        torch.set_num_threads(previous_threads)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_deterministic
