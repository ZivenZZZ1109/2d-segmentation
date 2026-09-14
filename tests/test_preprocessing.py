import unittest

import numpy as np

from seg2d.preprocessing import clahe_rgb, gamma_correct, match_histograms


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(7).integers(0, 256, (24, 32, 3), dtype=np.uint8)

    def test_gamma_identity(self):
        np.testing.assert_array_equal(gamma_correct(self.image, 1), self.image)

    def test_gamma_direction_and_endpoints(self):
        pixels = np.array([0, 64, 128, 255], dtype=np.uint8)[None, :, None].repeat(3, axis=2)
        bright = gamma_correct(pixels, 0.5)
        dark = gamma_correct(pixels, 2)
        self.assertGreater(bright[0, 1, 0], pixels[0, 1, 0])
        self.assertLess(dark[0, 1, 0], pixels[0, 1, 0])
        self.assertEqual(bright[0, 0, 0], 0)
        self.assertEqual(bright[0, -1, 0], 255)

    def test_invalid_parameters(self):
        for gamma in (0, -1, np.nan, np.inf):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                gamma_correct(self.image, gamma)
        with self.assertRaises(ValueError):
            clahe_rgb(self.image, tile_grid_size=(0, 8))

    def test_rejects_ambiguous_input_types(self):
        for image in (self.image.astype(float), self.image[:, :, 0], np.zeros((0, 3, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                gamma_correct(image)

    def test_histogram_identity(self):
        np.testing.assert_array_equal(match_histograms(self.image, self.image), self.image)

    def test_histogram_matches_constant_reference(self):
        reference = np.full((4, 5, 3), 96, dtype=np.uint8)
        np.testing.assert_array_equal(match_histograms(self.image, reference), np.full_like(self.image, 96))

    def test_operations_preserve_inputs_and_output_contract(self):
        before = self.image.copy()
        for output in (clahe_rgb(self.image), gamma_correct(self.image), match_histograms(self.image, self.image)):
            self.assertEqual(output.shape, self.image.shape)
            self.assertEqual(output.dtype, np.uint8)
        np.testing.assert_array_equal(self.image, before)

    def test_clahe_matches_direct_opencv_implementation(self):
        import cv2

        lab = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
        expected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        np.testing.assert_array_equal(clahe_rgb(self.image), expected)
