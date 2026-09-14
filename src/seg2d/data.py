"""Paired image/mask datasets with explicit splits and synchronized transforms."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .images import load_labels, load_rgb


def _index(directory: Path, extensions: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Missing directory: {directory}")
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            key = path.stem.casefold()
            if key in files:
                raise ValueError(f"Duplicate image/mask stem: {path.stem}")
            files[key] = path
    return files


def _digest(array: np.ndarray) -> str:
    digest = hashlib.sha256(str((array.shape, str(array.dtype))).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


class PairedDataset(Dataset):
    """Read flat images/ and masks/ directories; pair by case-insensitive stem."""

    def __init__(self, root: str | Path, num_classes: int, size: int = 64,
                 augment: bool = False, seed: int = 0):
        if type(num_classes) is not int or not 2 <= num_classes <= 256:
            raise ValueError("num_classes must be in [2, 256].")
        if type(size) is not int or size < 32:
            raise ValueError("Dataset size must be >= 32 for training-mode BatchNorm.")
        self.root = Path(root)
        self.num_classes, self.size, self.augment = num_classes, size, augment
        self.generator = torch.Generator().manual_seed(seed)
        images = _index(self.root / "images", {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
        masks = _index(self.root / "masks", {".png"})
        if not images or images.keys() != masks.keys():
            raise ValueError("Image/mask pairing must be nonempty and one-to-one; check missing or extra masks.")
        self.pairs = [(images[key], masks[key]) for key in sorted(images)]
        self.manifest = []
        for image_path, mask_path in self.pairs:
            image, mask = self._read(image_path, mask_path)
            self.manifest.append({
                "image": image_path.relative_to(self.root).as_posix(),
                "mask": mask_path.relative_to(self.root).as_posix(),
                "shape": list(mask.shape),
                "image_sha256": _digest(image),
                "mask_sha256": _digest(mask),
            })

    def _read(self, image_path, mask_path):
        image, mask = load_rgb(image_path), load_labels(mask_path)
        if mask.shape != image.shape[:2]:
            raise ValueError(f"Image/mask shape mismatch: {image_path.name}")
        if mask.min() < 0 or mask.max() >= self.num_classes:
            raise ValueError(f"Mask class indices outside [0, num_classes): {mask_path.name}")
        return image, mask.astype(np.uint8)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image, mask = self._read(*self.pairs[index])
        image = np.asarray(Image.fromarray(image).resize(
            (self.size, self.size), Image.Resampling.BILINEAR)).copy()
        mask = np.asarray(Image.fromarray(mask).resize(
            (self.size, self.size), Image.Resampling.NEAREST)).copy()
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        mask = torch.from_numpy(mask).long()
        if self.augment:
            turns = int(torch.randint(4, (), generator=self.generator))
            image, mask = torch.rot90(image, turns, (1, 2)), torch.rot90(mask, turns, (0, 1))
            if bool(torch.rand((), generator=self.generator) < 0.5):
                image, mask = image.flip(2), mask.flip(1)
        return image.contiguous(), mask.contiguous()


def require_disjoint(*datasets: PairedDataset) -> None:
    """Reject identical decoded images across splits, including renamed copies."""
    seen = set()
    for dataset in datasets:
        fingerprints = {record["image_sha256"] for record in dataset.manifest}
        if seen & fingerprints:
            raise ValueError("Identical decoded image content occurs across dataset splits.")
        seen.update(fingerprints)
