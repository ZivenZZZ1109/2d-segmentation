"""Explicit image and label conventions at file boundaries."""

from pathlib import Path

import numpy as np
from PIL import Image


def require_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an H x W x 3 uint8 RGB image.")
    if not image.shape[0] or not image.shape[1]:
        raise ValueError("Image must not be empty.")
    return image


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("RGB", "L"):
            raise ValueError("Only 8-bit RGB or grayscale input is supported; convert explicitly first.")
        return np.asarray(image.convert("RGB")).copy()


def load_labels(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("L", "P", "I", "I;16"):
            raise ValueError("Use a single-channel class-index mask, not a color visualization.")
        return np.asarray(image).copy()


def save_image(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def require_new_output(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    return path
