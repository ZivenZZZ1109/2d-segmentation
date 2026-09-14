"""Procedural example images; no research images or trained predictions are used."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .images import require_new_output, save_image
from .preprocessing import clahe_rgb, gamma_correct, match_histograms


def create_demo(output_dir: str | Path) -> Path:
    output_dir = require_new_output(output_dir)
    output_dir.mkdir(parents=True)
    width, height = 384, 256
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    texture = 5 * np.sin(50*x) * np.cos(35*y)
    image = np.stack((35+42*x+texture, 58+52*y+texture, 73+40*x+texture), axis=-1)
    image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    drawing = ImageDraw.Draw(image)
    drawing.ellipse((40, 50, 170, 185), fill=(111, 83, 70))
    drawing.rectangle((225, 110, 340, 215), fill=(65, 108, 87))
    original = np.asarray(image)
    reference = np.stack((60+140*x, 40+160*y, 90+90*x), axis=-1).astype(np.uint8)
    panels = [
        ("Input (procedural)", original),
        ("CLAHE (LAB luminance)", clahe_rgb(original)),
        ("Gamma = 0.65", gamma_correct(original, 0.65)),
        ("Histogram matching", match_histograms(original, reference)),
    ]
    canvas = Image.new("RGB", (2*width+48, 2*(height+42)+48), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for index, (label, array) in enumerate(panels):
        left = 16 + (index % 2) * (width+16)
        top = 16 + (index // 2) * (height+58)
        draw.text((left, top), label, fill=(20, 20, 20), font=font)
        canvas.paste(Image.fromarray(array), (left, top+30))
        save_image(array, output_dir / f"{index:02d}.png")
    save_image(reference, output_dir / "reference.png")
    path = output_dir / "preprocessing_demo.png"
    canvas.save(path)
    return path
