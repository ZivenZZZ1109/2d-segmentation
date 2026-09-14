from pathlib import Path
import unittest

from PIL import Image

from seg2d.images import load_labels


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
