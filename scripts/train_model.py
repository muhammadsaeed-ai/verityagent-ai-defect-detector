"""Train the PyTorch anomaly detector from defect-free image directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verityagent_defect_detector.training import TrainingConfig, train_autoencoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("data/train/good")); parser.add_argument("--validation-dir", type=Path, default=Path("data/validation/good")); parser.add_argument("--output", type=Path, default=Path("models/defect_autoencoder.pt")); parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--image-size", type=int, default=256); parser.add_argument("--latent-channels", type=int, default=16); parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--threshold-percentile", type=float, default=99.5); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu"); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); settings = TrainingConfig(image_size=args.image_size, latent_channels=args.latent_channels, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, threshold_percentile=args.threshold_percentile, seed=args.seed, device=args.device); result = train_autoencoder(args.train_dir, args.validation_dir, args.output, config=settings, overwrite=args.overwrite); print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
