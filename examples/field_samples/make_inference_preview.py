"""Display fixed field images, supplied annotations, and raw class-index predictions."""

import argparse
import math
from numbers import Real
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from seg2d.images import load_labels, load_rgb, require_new_output, require_rgb


ROOT = Path(__file__).resolve().parent
COLORS = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0)}
OVERLAY_ALPHA = 0.45


def paint_prediction(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Use annotation colors for foreground; leave class-zero image pixels visible."""
    image = require_rgb(image)
    labels = np.asarray(labels)
    if labels.shape != image.shape[:2] or labels.dtype.kind not in "iu":
        raise ValueError("Predictions must be aligned integer class-index masks.")
    if labels.min() < 0 or labels.max() > 3:
        raise ValueError("Predicted class IDs must be in [0, 3].")
    output = image.copy()
    for class_id, color in COLORS.items():
        output[labels == class_id] = color
    return output


def overlay_rgb(image: np.ndarray, colored: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """Blend a display image without decoding annotations or changing label geometry."""
    image, colored = require_rgb(image), require_rgb(colored)
    if colored.shape != image.shape:
        raise ValueError("Overlay inputs must have identical dimensions.")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Overlay alpha must be finite and in [0, 1].")
    return np.rint((1 - alpha) * image.astype(np.float64) + alpha * colored.astype(np.float64)).astype(np.uint8)


def create_preview(
    output: Path, predictions_dir: Path = ROOT / "predictions", view: str = "solid"
) -> None:
    if view not in {"solid", "overlay", "featured"}:
        raise ValueError("Preview view must be solid, overlay, or featured.")
    output = require_new_output(output)
    sample_numbers = (2,) if view == "featured" else (1, 2, 3)
    panels = []
    for number in sample_numbers:
        filename = f"sample_{number:02d}.png"
        image = load_rgb(ROOT / "images" / filename)
        annotation = load_rgb(ROOT / "annotations" / filename)
        if annotation.shape != image.shape:
            raise ValueError(f"Image and annotation dimensions differ: {filename}")
        prediction = paint_prediction(image, load_labels(predictions_dir / filename))
        if view != "solid":
            annotation = overlay_rgb(image, annotation)
            prediction = overlay_rgb(image, prediction)
        panels.append((image, annotation, prediction))

    height = 152 + len(panels) * 352
    canvas = Image.new("RGB", (1024, height), "white")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=22)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=16)
    heading = "Seagrass segmentation | Sample 02" if view == "featured" else "U-Net inference on real seafloor images"
    subtitle = (
        "Qualitative validation example | All three cases are retained in the example gallery"
        if view == "featured" else "Fixed examples | Corresponding validation-fold checkpoints"
    )
    draw.text((16, 12), heading, fill="black", font=title)
    draw.text((16, 42), subtitle, fill="black", font=small)
    for x, text, color in ((16, "Coral", COLORS[1]), (130, "Seagrass", COLORS[2]), (284, "Sea urchin", COLORS[3])):
        draw.rectangle((x, 68, x + 14, 82), fill=color, outline="black")
        draw.text((x + 22, 65), text, fill="black", font=small)
    draw.text((466, 65), "Others: original image texture", fill="black", font=small)
    headers = ("Image", "Supplied annotation", "U-Net prediction") if view == "solid" else (
        "Image", "Annotation (45% overlay)", "Prediction (45% overlay)"
    )
    for column, text in enumerate(headers):
        draw.text((16 + column * 336, 91), text, fill="black", font=font)
    for row, triple in enumerate(panels):
        top = 118 + row * 352
        for column, pixels in enumerate(triple):
            panel = Image.fromarray(pixels).resize((320, 320), Image.Resampling.NEAREST)
            canvas.paste(panel, (16 + column * 336, top))
        draw.text((16, top + 324), f"Sample {sample_numbers[row]:02d}", fill="black", font=small)
    footer = (
        "No test-time augmentation, smoothing, or manual prediction corrections."
        if view == "solid" else "Same opacity for annotations and predictions; original masks are unchanged."
    )
    draw.text((16, height - 26), footer, fill="black", font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "predictions")
    parser.add_argument("--view", choices=("solid", "overlay", "featured"), default="solid")
    args = parser.parse_args()
    create_preview(args.output, args.predictions_dir, args.view)
