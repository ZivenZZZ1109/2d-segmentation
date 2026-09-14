"""Strict checkpoint loading and deterministic evaluation-mode inference."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .images import require_rgb
from .model import UNet


def load_model(
    checkpoint: str | Path, num_classes: int, base_channels: int = 64, device: str = "cpu"
) -> UNet:
    if type(num_classes) is not int or not 2 <= num_classes <= 256:
        raise ValueError("num_classes must be in [2, 256] for uint8 PNG masks.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty state_dict or a dict containing state_dict.")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid or non-finite checkpoint tensor: {key}")
    model = UNet(n_classes=num_classes, base_channels=base_channels)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict(model: UNet, image: np.ndarray, size: int = 512) -> np.ndarray:
    """Resize to a square for inference, then restore label size with nearest sampling."""
    image = require_rgb(image)
    if type(size) is not int or size < 16:
        raise ValueError("size must be an integer >= 16.")
    if not 2 <= model.n_classes <= 256:
        raise ValueError("Prediction PNGs support 2 to 256 classes.")
    height, width = image.shape[:2]
    resized = np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)).copy()
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255
    tensor = tensor.to(next(model.parameters()).device)
    model.eval()
    logits = model(tensor)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Non-finite logits; refusing to convert them to class labels.")
    labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return np.asarray(Image.fromarray(labels).resize((width, height), Image.Resampling.NEAREST)).copy()
