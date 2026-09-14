"""Command-line entry points with explicit inputs and no implicit data discovery."""

import argparse
import json

from .images import load_labels, load_rgb, require_new_output, save_image
from .metrics import ConfusionMatrix
from .preprocessing import clahe_rgb, gamma_correct, match_histograms


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="seg2d", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate a procedural preprocessing example")
    demo.add_argument("--output-dir", default="runs/demo")
    toy = sub.add_parser("make-toy-data", help="Generate independent train/val/test shapes")
    toy.add_argument("--output-dir", default="runs/toy-data")
    toy.add_argument("--size", type=int, default=64)
    toy.add_argument("--seed", type=int, default=17)
    training_demo = sub.add_parser("train-demo", help="Train, test, and visualize a small CPU example")
    training_demo.add_argument("--output-dir", default="runs/train-demo")
    training_demo.add_argument("--epochs", type=int, default=15)
    training = sub.add_parser("train", help="Train U-Net with separate train/ and val/ directories")
    training.add_argument("data_dir")
    training.add_argument("output_dir")
    training.add_argument("--classes", type=int, required=True)
    training.add_argument("--size", type=int, default=64)
    training.add_argument("--base-channels", type=int, default=4)
    training.add_argument("--epochs", type=int, default=15)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=0.003)
    training.add_argument("--patience", type=int, default=8)
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    training.add_argument("--threads", type=int, default=1)
    training.add_argument("--no-augment", action="store_true")
    prep = sub.add_parser("preprocess", help="Process an 8-bit RGB image")
    prep.add_argument("input")
    prep.add_argument("output")
    prep.add_argument("--method", choices=("clahe", "gamma", "histogram"), required=True)
    prep.add_argument("--gamma", type=float, default=0.8)
    prep.add_argument("--clip-limit", type=float, default=2.0)
    prep.add_argument("--reference", help="Reference image required for histogram matching")
    evaluate = sub.add_parser("evaluate", help="Evaluate class-index PNG masks")
    evaluate.add_argument("target")
    evaluate.add_argument("prediction")
    evaluate.add_argument("--classes", type=int, required=True)
    predict_parser = sub.add_parser("predict", help="Run U-Net with your compatible checkpoint")
    predict_parser.add_argument("input")
    predict_parser.add_argument("checkpoint")
    predict_parser.add_argument("output")
    predict_parser.add_argument("--classes", type=int, required=True)
    predict_parser.add_argument("--base-channels", type=int, default=64)
    predict_parser.add_argument("--size", type=int, default=512)
    predict_parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import create_demo

            print(create_demo(args.output_dir))
        elif args.command == "make-toy-data":
            from .toy import create_toy_data

            print(create_toy_data(args.output_dir, size=args.size, seed=args.seed))
        elif args.command == "train-demo":
            from .toy import train_demo

            print(json.dumps(train_demo(args.output_dir, args.epochs), indent=2, allow_nan=False))
        elif args.command == "train":
            from .training import TrainConfig, train

            config = TrainConfig(
                num_classes=args.classes, size=args.size, base_channels=args.base_channels,
                epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                patience=args.patience, seed=args.seed, device=args.device, threads=args.threads,
                augment=not args.no_augment,
            )
            print(json.dumps(train(args.data_dir, args.output_dir, config), indent=2, allow_nan=False))
        elif args.command == "evaluate":
            metric = ConfusionMatrix(args.classes)
            metric.update(load_labels(args.target), load_labels(args.prediction))
            print(json.dumps(metric.compute(), indent=2, allow_nan=False))
        else:
            output = require_new_output(args.output)
            if output.suffix.lower() != ".png":
                raise ValueError("Output must be PNG to avoid lossy image or label storage.")
            image = load_rgb(args.input)
            if args.command == "predict":
                from .inference import load_model, predict

                model = load_model(args.checkpoint, args.classes, args.base_channels, args.device)
                result = predict(model, image, args.size)
            elif args.method == "clahe":
                result = clahe_rgb(image, clip_limit=args.clip_limit)
            elif args.method == "gamma":
                result = gamma_correct(image, args.gamma)
            else:
                if not args.reference:
                    raise ValueError("--reference is required for histogram matching.")
                result = match_histograms(image, load_rgb(args.reference))
            save_image(result, output)
            print(output)
    except (ValueError, OSError, RuntimeError, ImportError, FloatingPointError) as error:
        parser.exit(2, f"seg2d: {error}\n")
