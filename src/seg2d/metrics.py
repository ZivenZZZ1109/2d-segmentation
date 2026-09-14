"""Pixel-pooled segmentation metrics without retaining all predicted pixels."""

import numpy as np


class ConfusionMatrix:
    """Rows are ground truth, columns predictions; absent-class metrics are zero."""

    def __init__(self, num_classes: int):
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2.")
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        target, prediction = np.asarray(target), np.asarray(prediction)
        if target.shape != prediction.shape or target.size == 0:
            raise ValueError("Target and prediction must have the same nonempty shape.")
        for array in (target, prediction):
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError("Labels must be integer class indices, not logits or probabilities.")
            if array.min() < 0 or array.max() >= self.num_classes:
                raise ValueError("Class indices are outside [0, num_classes).")
        indices = self.num_classes * target.astype(np.int64).ravel() + prediction.astype(np.int64).ravel()
        self.matrix += np.bincount(indices, minlength=self.num_classes**2).reshape(self.matrix.shape)

    def compute(self) -> dict:
        if self.matrix.sum() == 0:
            raise ValueError("No pixels have been accumulated.")
        true_positive = np.diag(self.matrix).astype(np.float64)
        support = self.matrix.sum(axis=1)
        predicted = self.matrix.sum(axis=0)

        def divide(numerator, denominator):
            return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

        iou = divide(true_positive, support + predicted - true_positive)
        f1 = divide(2 * true_positive, support + predicted)
        return {
            "pixel_accuracy": float(true_positive.sum() / support.sum()),
            "mean_iou": float(iou.mean()),
            "macro_f1": float(f1.mean()),
            "iou_per_class": iou.tolist(),
            "f1_per_class": f1.tolist(),
            "precision_per_class": divide(true_positive, predicted).tolist(),
            "recall_per_class": divide(true_positive, support).tolist(),
            "support_per_class": support.tolist(),
            "confusion_matrix": self.matrix.tolist(),
            "absent_class_policy": "zero; macro averages include all configured classes",
        }
