"""Display fixed field images, supplied annotations, and raw class-index predictions."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from seg2d.images import load_labels, load_rgb, require_new_output, require_rgb


ROOT = Path(__file__).resolve().parent
COLORS = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0)}


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


def create_preview(output: Path, predictions_dir: Path = ROOT / "predictions") -> None:
    output = require_new_output(output)
    panels = []
    for number in range(1, 4):
        filename = f"sample_{number:02d}.png"
        image = load_rgb(ROOT / "images" / filename)
        annotation = load_rgb(ROOT / "annotations" / filename)
        if annotation.shape != image.shape:
            raise ValueError(f"Image and annotation dimensions differ: {filename}")
        prediction = paint_prediction(image, load_labels(predictions_dir / filename))
        panels.append((image, annotation, prediction))

    canvas = Image.new("RGB", (1024, 1208), "white")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=22)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=16)
    draw.text((16, 12), "U-Net inference on real seafloor images", fill="black", font=title)
    draw.text((16, 42), "Fixed examples | Corresponding validation-fold checkpoints", fill="black", font=small)
    for x, text, color in ((16, "Coral", COLORS[1]), (130, "Seagrass", COLORS[2]), (284, "Sea urchin", COLORS[3])):
        draw.rectangle((x, 68, x + 14, 82), fill=color, outline="black")
        draw.text((x + 22, 65), text, fill="black", font=small)
    draw.text((466, 65), "Others: original image texture", fill="black", font=small)
    for column, text in enumerate(("Image", "Supplied annotation", "U-Net prediction")):
        draw.text((16 + column * 336, 91), text, fill="black", font=font)
    for row, triple in enumerate(panels):
        top = 118 + row * 352
        for column, pixels in enumerate(triple):
            panel = Image.fromarray(pixels).resize((320, 320), Image.Resampling.NEAREST)
            canvas.paste(panel, (16 + column * 336, top))
        draw.text((16, top + 324), f"Sample {row + 1:02d}", fill="black", font=small)
    draw.text((16, 1182), "No test-time augmentation, smoothing, or manual prediction corrections.", fill="black", font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "predictions")
    args = parser.parse_args()
    create_preview(args.output, args.predictions_dir)
