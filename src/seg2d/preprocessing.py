"""Deterministic preprocessing for uint8 RGB images; inputs are never mutated."""

import numpy as np

from .images import require_rgb


def gamma_correct(image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """Apply y = x**gamma on [0, 1]. Values below one brighten the image."""
    image = require_rgb(image)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive.")
    lookup = np.rint(255 * (np.arange(256, dtype=np.float64) / 255) ** gamma)
    return lookup.astype(np.uint8)[image]


def clahe_rgb(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply CLAHE to the L channel in OpenCV's 8-bit LAB representation."""
    import cv2

    image = require_rgb(image)
    if not np.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("clip_limit must be finite and positive.")
    if len(tile_grid_size) != 2 or any(type(n) is not int or n < 1 for n in tile_grid_size):
        raise ValueError("tile_grid_size must contain two positive integers.")
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    operator = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    lab[:, :, 0] = operator.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def match_histograms(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match each RGB channel's empirical CDF to a reference, independently."""
    image, reference = require_rgb(image), require_rgb(reference)
    result = np.empty_like(image)
    for channel in range(3):
        source = image[:, :, channel]
        _, inverse, counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
        values, ref_counts = np.unique(reference[:, :, channel], return_counts=True)
        quantiles = np.cumsum(counts, dtype=np.float64) / source.size
        ref_quantiles = np.cumsum(ref_counts, dtype=np.float64) / reference[:, :, channel].size
        matched = np.interp(quantiles, ref_quantiles, values)
        result[:, :, channel] = np.rint(matched[inverse]).astype(np.uint8).reshape(source.shape)
    return result
