import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from seg2d.cli import main
from seg2d.images import load_labels, load_rgb


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / 'input.png'
        Image.fromarray(np.full((24, 32, 3), 60, dtype=np.uint8)).save(self.input)

    def invoke(self, args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(args)
        return output.getvalue()

    def test_gamma_command(self):
        output = self.root / 'output.png'
        self.invoke(['preprocess', str(self.input), str(output), '--method', 'gamma'])
        self.assertGreater(load_rgb(output).mean(), 60)

    def test_no_overwrite(self):
        original = self.input.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(['preprocess', str(self.input), str(self.input), '--method', 'clahe'])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(self.input.read_bytes(), original)

    def test_histogram_requires_reference(self):
        output = self.root / 'output.png'
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['preprocess', str(self.input), str(output), '--method', 'histogram'])
        self.assertFalse(output.exists())

    def test_metrics_command_and_palette_labels(self):
        labels = self.root / 'labels.png'
        Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8)).convert('P').save(labels)
        np.testing.assert_array_equal(load_labels(labels), [[0, 1], [1, 0]])
        result = json.loads(self.invoke(['evaluate', str(labels), str(labels), '--classes', '2']))
        self.assertEqual(result['mean_iou'], 1)

    def test_rgb_labels_rejected(self):
        with self.assertRaises(ValueError):
            load_labels(self.input)

    def test_demo_is_reproducible(self):
        first, second = self.root / 'first', self.root / 'second'
        self.invoke(['demo', '--output-dir', str(first)])
        self.invoke(['demo', '--output-dir', str(second)])
        self.assertEqual((first / 'preprocessing_demo.png').read_bytes(), (second / 'preprocessing_demo.png').read_bytes())
