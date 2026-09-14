from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install the model extra to run PyTorch tests.')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        from seg2d.model import UNet

        torch.manual_seed(11)
        self.model = UNet(n_classes=3, base_channels=4).eval()

    def test_even_and_odd_dimensions(self):
        with torch.inference_mode():
            for height, width in ((32, 32), (35, 49)):
                logits = self.model(torch.zeros(1, 3, height, width))
                self.assertEqual(tuple(logits.shape), (1, 3, height, width))
                self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_backward_gradients_are_finite(self):
        self.model.train()
        logits = self.model(torch.rand(2, 3, 32, 32))
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(2, 32, 32, dtype=torch.long))
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()))

    def test_checkpoint_round_trip_and_prediction(self):
        from seg2d.inference import load_model, predict

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'
            torch.save({'state_dict': self.model.state_dict()}, path)
            loaded = load_model(path, num_classes=3, base_channels=4)
            for key, expected in self.model.state_dict().items():
                self.assertTrue(torch.equal(expected, loaded.state_dict()[key]))
            image = np.zeros((40, 53, 3), dtype=np.uint8)
            first, second = predict(loaded, image, size=32), predict(loaded, image, size=32)
            self.assertEqual(first.shape, (40, 53))
            self.assertLess(int(first.max()), 3)
            np.testing.assert_array_equal(first, second)

    def test_nonfinite_bn_buffer_rejected(self):
        from seg2d.inference import load_model

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.pt'
            state = self.model.state_dict()
            state['inc.double_conv.1.running_mean'][0] = float('nan')
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                load_model(path, 3, 4)

    def test_nonfinite_predictions_rejected(self):
        from seg2d.inference import predict

        with torch.no_grad():
            self.model.outc.bias.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            predict(self.model, np.zeros((32, 32, 3), dtype=np.uint8), size=32)

    def test_invalid_spatial_size_rejected(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 3, 8, 8))
