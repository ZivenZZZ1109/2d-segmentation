# Design Notes

## Separation Of Concerns

Preprocessing, model execution, and metrics are independent. Importing `seg2d` or
running a preprocessing command does not initialize PyTorch or allocate GPU memory.
Paths come from command-line arguments; no module scans an external project tree.
The default inference and training device is CPU. GPU use must be requested explicitly.

The generic trainer is separate from inference and preprocessing. Its protocol and
data requirements are documented in [training.md](training.md). It does not import
or execute an external research training pipeline.

## Array Contracts

- Preprocessing accepts nonempty H x W x 3 arrays with dtype uint8 and RGB order.
- Model inputs are floating-point N x 3 x H x W tensors with image intensities in [0, 1].
- Labels are integer arrays. Resizing labels uses nearest-neighbor interpolation.
- Inference uses a square input size and restores the original output dimensions.
- CLI image outputs are PNG. Existing files and demo directories are not overwritten.

## U-Net

The encoder has widths C, 2C, 4C, 8C, and 16C. Each block contains two 3x3
convolution / batch-normalization / ReLU sequences. Max pooling downsamples between
blocks. The decoder uses nearest-neighbor upsampling, concatenated skip features,
and the same two-convolution blocks. A 1x1 convolution produces class logits.

The default C is 64. Smaller widths are useful for CPU tests but define different
checkpoints. The implementation preserves the local baseline's layer names and
channel layout at the default width. It is not a byte-for-byte release of a complete
historical training pipeline. Odd spatial dimensions are supported; H and W must
be at least 16. During training, BatchNorm also requires more than one value per
channel at the bottleneck (for example, use batch size >= 2 at 16x16 resolution).

## Metrics

The confusion matrix has ground-truth classes on rows and predicted classes on
columns. Counts accumulate across images before ratios are computed, so the result
is pixel-pooled, not a mean of per-image scores. Persistent metric memory is O(C^2);
one temporary encoded-index array is used per update.

For each class, TP is the diagonal count, FP is the column sum minus TP, and FN is
the row sum minus TP. IoU is TP/(TP+FP+FN); F1 is 2TP/(2TP+FP+FN). Undefined ratios
are zero. Macro scores average over all configured classes. Background is included
if it is one of those classes. Invalid labels, shape mismatches, empty inputs, and
empty accumulators fail explicitly. There is no automatic ignore-index behavior.

## Verification Scope

Tests cover known confusion matrices, pixel-pooling equivalence, absent classes,
gamma identity and direction, histogram identity, direct OpenCV CLAHE equivalence,
input preservation, CLI validation, reproducible examples, checkpoint round trips,
finite gradients, non-finite checkpoints/logits, and odd model input sizes.
Training tests additionally check pairing errors, split overlap, synchronized
transforms, repeatability, best-checkpoint selection, failure records, and atomic
checkpoint writes. CI runs a short end-to-end training demo on generated data.

The procedural demo needs no research data and has no scientific interpretation.
CI is configured for CPU execution; a workflow file alone is not evidence that a
remote run has passed. No latency, throughput, or segmentation-accuracy benchmark
is claimed here.
