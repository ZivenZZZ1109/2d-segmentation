import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from seg2d.images import load_labels


ROOT = Path(__file__).resolve().parents[1] / "examples" / "field_samples"
SPEC = importlib.util.spec_from_file_location("field_preview", ROOT / "make_inference_preview.py")
PREVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIEW)


class FieldExampleTests(unittest.TestCase):
    def test_only_three_metadata_free_pairs_are_bundled(self):
        root = Path(__file__).resolve().parents[1] / "examples" / "field_samples"
        for folder in ("images", "annotations"):
            paths = sorted((root / folder).glob("*.png"))
            self.assertEqual([path.name for path in paths], [f"sample_{n:02d}.png" for n in range(1, 4)])
            for path in paths:
                with Image.open(path) as handle:
                    self.assertEqual(handle.mode, "RGB")
                    self.assertEqual(handle.size, (512, 512))
                    self.assertFalse(handle.info)
                    self.assertFalse(handle.getexif())
        with self.assertRaisesRegex(ValueError, "class-index"):
            load_labels(root / "annotations" / "sample_01.png")

    def test_prediction_manifest_and_masks(self):
        manifest = json.loads((ROOT / "inference.json").read_text())
        self.assertEqual(manifest["training_seed"], 42)
        self.assertEqual(manifest["num_classes"], 4)
        self.assertEqual([row["validation_fold"] for row in manifest["samples"]], [3, 3, 1])
        self.assertEqual([row["sample"] for row in manifest["samples"]], [f"sample_{n:02d}" for n in range(1, 4)])
        self.assertEqual(len(list((ROOT / "predictions").glob("*.png"))), 3)
        for row in manifest["samples"]:
            name = row["sample"] + ".png"
            for folder, key in (("images", "image_sha256"), ("predictions", "prediction_sha256")):
                self.assertEqual(hashlib.sha256((ROOT / folder / name).read_bytes()).hexdigest(), row[key])
            with Image.open(ROOT / "predictions" / name) as handle:
                self.assertEqual(handle.mode, "L")
                self.assertEqual(handle.size, (512, 512))
                self.assertFalse(handle.info)
                self.assertFalse(handle.getexif())
                labels = np.asarray(handle)
                self.assertGreaterEqual(int(labels.min()), 0)
                self.assertLessEqual(int(labels.max()), 3)

    def test_prediction_colors_do_not_modify_inputs(self):
        image = np.full((2, 2, 3), 71, dtype=np.uint8)
        labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        result = PREVIEW.paint_prediction(image, labels)
        np.testing.assert_array_equal(image, np.full_like(image, 71))
        np.testing.assert_array_equal(labels, [[0, 1], [2, 3]])
        np.testing.assert_array_equal(result, [[[71, 71, 71], [255, 0, 0]], [[0, 255, 0], [255, 255, 0]]])

    def test_preview_rejects_invalid_labels(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for labels in (np.zeros((3, 2), dtype=np.uint8), np.zeros((2, 2)), np.full((2, 2), -1), np.full((2, 2), 4)):
            with self.assertRaises(ValueError):
                PREVIEW.paint_prediction(image, labels)

    def test_inference_preview_layout_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preview.png"
            repeated = Path(directory) / "repeated.png"
            PREVIEW.create_preview(output)
            PREVIEW.create_preview(repeated)
            with Image.open(output) as actual, Image.open(repeated) as expected:
                self.assertEqual(actual.size, (1024, 1208))
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            with Image.open(ROOT / "inference_preview.png") as bundled:
                self.assertEqual(bundled.size, (1024, 1208))
                self.assertFalse(bundled.info)
            with self.assertRaises(FileExistsError):
                PREVIEW.create_preview(output)

    def test_equal_opacity_overlay_preserves_inputs_and_background(self):
        image = np.full((2, 2, 3), 80, dtype=np.uint8)
        labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        colored = PREVIEW.paint_prediction(image, labels)
        original_colored = colored.copy()
        actual = PREVIEW.overlay_rgb(image, colored)
        expected = np.rint(0.55 * image.astype(float) + 0.45 * colored.astype(float)).astype(np.uint8)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[0, 0], image[0, 0])
        np.testing.assert_array_equal(image, np.full_like(image, 80))
        np.testing.assert_array_equal(colored, original_colored)
        np.testing.assert_array_equal(PREVIEW.overlay_rgb(image, colored, 0), image)
        np.testing.assert_array_equal(PREVIEW.overlay_rgb(image, colored, 1), colored)

    def test_overlay_rejects_invalid_parameters(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for alpha in (-0.1, 1.1, float("nan"), float("inf"), True, "0.45"):
            with self.assertRaises(ValueError):
                PREVIEW.overlay_rgb(image, image, alpha)
        with self.assertRaises(ValueError):
            PREVIEW.overlay_rgb(image, np.zeros((3, 2, 3), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                PREVIEW.create_preview(Path(directory) / "unused.png", view="unknown")

    def test_overlay_and_featured_views_use_original_panels(self):
        from seg2d.images import load_rgb

        with tempfile.TemporaryDirectory() as directory:
            for view, height, name in (("overlay", 1208, "inference_overlay.png"), ("featured", 504, "featured_inference.png")):
                output = Path(directory) / name
                PREVIEW.create_preview(output, view=view)
                with Image.open(output) as rendered:
                    self.assertEqual(rendered.size, (1024, height))
                    sample = "sample_02.png" if view == "featured" else "sample_01.png"
                    image = load_rgb(ROOT / "images" / sample)
                    annotation = load_rgb(ROOT / "annotations" / sample)
                    predicted = PREVIEW.paint_prediction(image, load_labels(ROOT / "predictions" / sample))
                    panels = (image, PREVIEW.overlay_rgb(image, annotation), PREVIEW.overlay_rgb(image, predicted))
                    for column, panel in enumerate(panels):
                        expected = Image.fromarray(panel).resize((320, 320), Image.Resampling.NEAREST)
                        actual = rendered.crop((16 + 336 * column, 118, 336 + 336 * column, 438))
                        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
                with Image.open(ROOT / name) as bundled:
                    self.assertEqual(bundled.size, (1024, height))
                    self.assertFalse(bundled.info)
                    self.assertFalse(bundled.getexif())
