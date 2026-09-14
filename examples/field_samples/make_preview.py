"""Contact sheet of the bundled field examples; annotation pixels are not predictions."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_preview(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    root = Path(__file__).resolve().parent
    canvas = Image.new("RGB", (792, 1260), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default(size=18)
    draw.text((16, 12), "Field images and annotations (not model predictions)", fill="black", font=font)
    draw.text((16, 45), "Image", fill="black", font=font)
    draw.text((408, 45), "Original RGB annotation", fill="black", font=font)
    for row in range(3):
        top = 78 + row * 394
        for column, folder in enumerate(("images", "annotations")):
            with Image.open(root / folder / f"sample_{row + 1:02d}.png") as handle:
                sample = handle.convert("RGB")
                sample.thumbnail((368, 368), Image.Resampling.NEAREST)
                canvas.paste(sample, (16 + column * 392, top))
        draw.text((16, top + 370), f"Sample {row + 1:02d}", fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    create_preview(parser.parse_args().output)
