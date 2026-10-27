"""Command-line interface for the VerityAgent defect detector."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from .comparison import compare_images
from .config import ComparisonConfig
from .inference import predict_with_checkpoint
from .reporting import save_anomaly_output, save_comparison_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verity-defect", description="Inspect product images with OpenCV comparison or a trained PyTorch model.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare", help="Compare inspection image against a known-good reference")
    compare.add_argument("--reference", required=True, type=Path); compare.add_argument("--inspection", required=True, type=Path); compare.add_argument("--output-dir", required=True, type=Path)
    compare.add_argument("--threshold", type=int, default=25); compare.add_argument("--blur-kernel", type=int, default=3); compare.add_argument("--min-area", type=int, default=100); compare.add_argument("--align", action="store_true"); compare.add_argument("--mode", choices=("grayscale", "color"), default="grayscale"); compare.add_argument("--no-illumination-normalization", action="store_true")
    infer = subparsers.add_parser("infer", help="Run trained-model anomaly detection")
    infer.add_argument("--checkpoint", required=True, type=Path); infer.add_argument("--inspection", required=True, type=Path); infer.add_argument("--output-dir", required=True, type=Path); infer.add_argument("--min-area", type=int, default=100); infer.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser(); args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            result = compare_images(args.reference, args.inspection, config=ComparisonConfig(threshold=args.threshold, blur_kernel=args.blur_kernel, min_area=args.min_area, align=args.align, mode=args.mode, normalize_illumination=not args.no_illumination_normalization))
            outputs = save_comparison_output(args.output_dir, result)
        elif args.command == "infer":
            result = predict_with_checkpoint(args.checkpoint, args.inspection, min_area=args.min_area, device=args.device)
            outputs = save_anomaly_output(args.output_dir, result)
        else:
            parser.error("Unsupported command"); return 2
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr); return 2
    print(f"Inspection complete. Method: {args.command}")
    for name, path in outputs.items(): print(f"{name}: {Path(args.output_dir) / path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
