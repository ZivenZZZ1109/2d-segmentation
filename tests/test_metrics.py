import unittest

import numpy as np

from seg2d.metrics import ConfusionMatrix


class MetricTests(unittest.TestCase):
    def test_hand_calculated_two_class_case(self):
        metric = ConfusionMatrix(2)
        metric.update(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        result = metric.compute()
        self.assertEqual(result['confusion_matrix'], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result['pixel_accuracy'], 0.75)
        self.assertAlmostEqual(result['mean_iou'], (0.5 + 2/3) / 2)
        np.testing.assert_allclose(result['f1_per_class'], [2/3, 0.8])

    def test_incremental_equals_pooled(self):
        target = np.array([0, 1, 2, 0, 1, 2])
        prediction = np.array([0, 1, 1, 1, 2, 2])
        pooled, streamed = ConfusionMatrix(3), ConfusionMatrix(3)
        pooled.update(target, prediction)
        streamed.update(target[:2], prediction[:2])
        streamed.update(target[2:], prediction[2:])
        self.assertEqual(pooled.compute(), streamed.compute())

    def test_absent_classes_are_explicitly_zero(self):
        metric = ConfusionMatrix(4)
        metric.update(np.array([0, 0]), np.array([0, 0]))
        self.assertEqual(metric.compute()['iou_per_class'], [1, 0, 0, 0])
        self.assertEqual(metric.compute()['mean_iou'], 0.25)

    def test_unsigned_64_bit_indices(self):
        metric = ConfusionMatrix(2)
        metric.update(np.array([0, 1], dtype=np.uint64), np.array([0, 1], dtype=np.uint64))
        self.assertEqual(metric.compute()['mean_iou'], 1)

    def test_invalid_labels(self):
        for target, prediction in (([0], [2]), ([-1], [0]), ([0.0], [0]), ([0, 1], [0]), ([], [])):
            with self.subTest(target=target), self.assertRaises(ValueError):
                ConfusionMatrix(2).update(np.array(target), np.array(prediction))

    def test_empty_accumulator(self):
        with self.assertRaises(ValueError):
            ConfusionMatrix(2).compute()
