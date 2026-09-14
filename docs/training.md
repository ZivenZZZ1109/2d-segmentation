# Training Guide

## Run The Example

```bash
python -m pip install -e '.[model]'
python -m seg2d train-demo --output-dir runs/train-demo
```

The example uses three classes: background (0), circle (1), and rectangle (2).
Images are generated independently for train, validation, and test using separate
NumPy random streams derived from seed 17. Noise, object positions, and colors vary.
The split counts are 48/12/12, and images are 64x64. The shapes and color cues make
this an easy software example, not a meaningful computer-vision benchmark.

A four-level U-Net with base width 4 is trained from scratch with ordinary
cross-entropy and Adam. Defaults: learning rate 0.003, batch size 8, 15 epochs,
patience 8, training seed 7, one CPU thread. Augmentation applies synchronized
quarter-turn rotations and horizontal flips to training images and masks only.
Validation and test transforms are fixed. The test split is evaluated after
checkpoint selection and does not influence early stopping.

The displayed preview uses test samples 000-003 in filename order. Test metrics
use all 12 test images, include background, and use the class conventions described
in [design.md](design.md). Preview masks are actual model outputs. There is no
image selection by accuracy, postprocessing, or transfer learning.

## Directory Contract

```text
my_dataset/
  train/
    images/    sample_001.jpg, sample_002.png, ...
    masks/     sample_001.png, sample_002.png, ...
  val/
    images/    sample_101.jpg, ...
    masks/     sample_101.png, ...
```

Directories are flat. Image extensions are PNG, JPG, JPEG, TIF, and TIFF; images
must decode as 8-bit RGB or grayscale. Masks are PNG class indices, not RGB
visualizations. Palette PNGs are interpreted as indices. Every supported image
must have exactly one mask with the same case-insensitive stem and original shape.
Duplicate stems, missing/extra masks, unsupported image modes, and labels outside
`[0, classes)` are errors. Other file extensions are ignored. Alpha or RGB-label
conversion must be explicit before training.

Files are validated at startup, then loaded lazily during training. Keep the input
files unchanged during a run. Decoded-image SHA-256 hashes detect identical images
across splits, including copies with different filenames. This does **not** detect
near-duplicates, overlapping crops, or related subjects. Choose subject-, scene-,
or time-separated splits yourself when the application requires them. The trainer
does not automatically partition data or inspect a `test/` directory.

Images are resized bilinearly to a square; masks use nearest-neighbor sampling.
The minimum training size is 32 to keep the BatchNorm bottleneck valid even for
a final batch of one image. No normalization beyond division by 255 is applied.
There is no ignore label or automatic class weighting. Evaluation within the
trainer uses the resized grid; the standalone `predict` command restores the
original mask size. These are different evaluation conventions for nonsquare or
differently sized images.

## Commands

Generate data without training:

```bash
python -m seg2d make-toy-data --output-dir runs/toy-data --seed 17
```

Run the generic trainer and evaluate one held-out image:

```bash
python -m seg2d train runs/toy-data runs/toy-run --classes 3 --size 64 --base-channels 4 --epochs 15 --seed 7
python -m seg2d predict runs/toy-data/test/images/000.png runs/toy-run/best.pt runs/test-000.png --classes 3 --size 64 --base-channels 4
python -m seg2d evaluate runs/toy-data/test/masks/000.png runs/test-000.png --classes 3
```

Use `train --help` for learning rate, batch size, patience, thread count, and
`--no-augment`. CUDA requires `--device cuda`; FP16/AMP is intentionally not part
of this minimal trainer. The model and resize settings must match at inference.
All output directories must be new. No existing run is silently resumed or replaced.

## Artifacts And Failure Behavior

| File | Contents |
| --- | --- |
| `config.json` | Effective configuration |
| `environment.json` | Python/PyTorch/NumPy versions, device and precision |
| `split_manifest.json` | Relative sample paths, shapes, decoded image/mask hashes |
| `history.csv` | Per-epoch pixel-averaged losses and validation mIoU |
| `best.pt` | Model state, configuration, epoch and best validation loss |
| `report.json` | Best-checkpoint validation metrics and elapsed training/evaluation time |
| `status.json` | Completed or failed state; errors are not converted into scores |

The selected checkpoint minimizes finite validation cross-entropy. Equal losses
are not improvements. Patience counts consecutive non-improving epochs. Best
weights are saved immediately rather than retaining references to mutable tensors.
Temporary files and atomic replacement protect the previous checkpoint if a new
write fails. This is not a guarantee against all filesystem or power-loss failures.

Non-finite logits, losses, gradients, parameters, and BatchNorm buffers stop the
run. The last healthy checkpoint, if one exists, is retained, but a failed run is
not reported as completed. The checkpoint stores inference state, **not** optimizer
or RNG state; exact training resumption is not supported. External termination can
leave incomplete artifacts, which must not be treated as a completed report.

Seeds initialize Python, NumPy, PyTorch, shuffle order and training transforms.
Loading uses zero workers; cuDNN benchmarking is disabled. The CLI is intended for
one run per process. Calling the Python training API also resets the process's RNG
streams. CPU tests check repeatability in the same environment, not bitwise
equivalence between machines, devices or framework versions. CUDA has not been
validated by the CPU CI workflow. See [PyTorch's reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
and [checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

Research images, trained research models, and original experiment configurations
are not required by this training workflow. Three separate field image examples
are included for viewing only, not used in this workflow. If you substitute your
own data, obtain permission before redistributing examples or derived visualizations.
