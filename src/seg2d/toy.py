"""Generated shapes for an end-to-end software example, not a research benchmark."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .images import require_new_output, save_image


PALETTE = np.array([[28, 36, 48], [225, 135, 65], [58, 165, 145]], dtype=np.uint8)


def create_toy_data(output_dir: str | Path, size: int = 64, seed: int = 17,
                    counts: tuple[int, int, int] = (48, 12, 12)) -> Path:
    if type(size) is not int or size < 32:
        raise ValueError("Toy size must be >= 32.")
    if len(counts) != 3 or any(type(count) is not int or count < 1 for count in counts):
        raise ValueError("Provide positive train, validation and test image counts.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    root = require_new_output(output_dir)
    root.mkdir(parents=True)
    streams = np.random.SeedSequence(seed).spawn(3)
    for split, count, stream in zip(("train", "val", "test"), counts, streams):
        rng = np.random.default_rng(stream)
        for index in range(count):
            mask_image = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(mask_image)
            radius = int(rng.integers(size // 8, size // 5))
            x, y = rng.integers(radius, size - radius, size=2)
            draw.ellipse((int(x-radius), int(y-radius), int(x+radius), int(y+radius)), fill=1)
            width, height = rng.integers(size // 5, size // 3, size=2)
            left, top = rng.integers(0, size - max(width, height), size=2)
            draw.rectangle((int(left), int(top), int(left+width), int(top+height)), fill=2)
            mask = np.asarray(mask_image)
            colors = PALETTE.astype(float) + rng.uniform(-15, 15, (3, 3))
            image = colors[mask] + rng.normal(0, 5, (size, size, 3))
            name = f"{index:03d}.png"
            save_image(np.clip(image, 0, 255).astype(np.uint8), root / split / "images" / name)
            save_image(mask, root / split / "masks" / name)
    metadata = {"description": "Procedural shapes; no field data or scientific benchmark.",
                "seed": seed, "size": size, "classes": ["background", "circle", "rectangle"],
                "counts": dict(zip(("train", "val", "test"), counts))}
    (root / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return root


def train_demo(output_dir: str | Path, epochs: int = 15) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from .data import PairedDataset, require_disjoint
    from .images import load_labels, load_rgb
    from .inference import load_model, predict
    from .training import TrainConfig, evaluate, train, write_json

    config = TrainConfig(epochs=epochs)
    config.validate()
    root = require_new_output(output_dir)
    root.mkdir(parents=True)
    data = create_toy_data(root / "data")
    splits = [PairedDataset(data / split, 3) for split in ("train", "val", "test")]
    require_disjoint(*splits)
    report = train(data, root / "training", config)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.threads)
        model = load_model(root / "training" / "best.pt", 3, config.base_channels)
        test = evaluate(model, DataLoader(splits[2], batch_size=config.batch_size), 3, "cpu")
        test["scope"] = "Independent generated test split; not evidence of real-world accuracy."
        write_json(root / "test_metrics.json", test)
        write_json(root / "test_manifest.json", {"test": splits[2].manifest})
        canvas = Image.new("RGB", (640, 4 * 206 + 75), "white")
        draw, font = ImageDraw.Draw(canvas), ImageFont.load_default(size=16)
        draw.text((16, 10), "Generated test shapes: first four samples, not selected by score", fill="black", font=font)
        for column, title in enumerate(("Input", "Ground truth", "Prediction")):
            draw.text((16 + column * 208, 43), title, fill="black", font=font)
        for row, (image_path, mask_path) in enumerate(splits[2].pairs[:4]):
            image, target = load_rgb(image_path), load_labels(mask_path)
            labels = predict(model, image, size=config.size)
            save_image(labels, root / "predictions" / image_path.name)
            for column, array in enumerate((image, PALETTE[target], PALETTE[labels])):
                canvas.paste(Image.fromarray(array).resize((192, 192), Image.Resampling.NEAREST),
                             (16 + column * 208, 75 + row * 206))
        canvas.save(root / "predictions.png")
        return {"output_dir": str(root), "best_epoch": report["best_epoch"],
                "test_mean_iou": test["mean_iou"], "scope": test["scope"]}
    finally:
        torch.set_num_threads(previous_threads)
