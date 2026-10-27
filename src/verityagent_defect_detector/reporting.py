"""Safe JSON reporting and image artifact export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .comparison import ComparisonResult
from .inference import AnomalyResult


def _prepare_output_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser()
    if directory.exists():
        if not directory.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {directory}")
        if any(directory.iterdir()):
            raise FileExistsError(f"Output directory is not empty; refusing to overwrite files: {directory}")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    return directory


def _write_image(path: Path, image: np.ndarray) -> None:
    if image.size == 0 or not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write result image: {path.name}")


def _write_artifacts(directory: Path, report: dict[str, Any], images: dict[str, np.ndarray]) -> dict[str, str]:
    saved: dict[str, str] = {}
    for name, image in images.items():
        path = directory / f"{name}.png"
        _write_image(path, image)
        saved[name] = path.name
    report["artifacts"] = saved
    report_path = directory / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    saved["report"] = report_path.name
    return saved


def save_comparison_output(output_dir: str | Path, result: ComparisonResult) -> dict[str, str]:
    """Save reference, inspection, difference, mask, heatmap, overlay, and JSON."""
    directory = _prepare_output_directory(output_dir)
    return _write_artifacts(directory, result.to_report(), {"reference": result.reference_image, "inspection": result.inspection_image, "difference_map": result.difference_map, "binary_mask": result.binary_mask, "anomaly_heatmap": result.heatmap, "overlay": result.overlay, "annotated_inspection": result.annotated_image})


def save_anomaly_output(output_dir: str | Path, result: AnomalyResult) -> dict[str, str]:
    """Save trained-model anomaly maps, visualizations, and JSON report."""
    directory = _prepare_output_directory(output_dir)
    return _write_artifacts(directory, result.to_report(), {"inspection": result.inspection_image, "anomaly_map": np.clip(result.anomaly_map * 255, 0, 255).astype(np.uint8), "binary_mask": result.binary_mask, "anomaly_heatmap": result.heatmap, "overlay": result.overlay, "annotated_inspection": result.annotated_image})


def write_json(path: str | Path, payload: dict[str, Any], *, overwrite: bool = False) -> Path:
    """Write a standalone JSON object without overwriting by default."""
    destination = Path(path).expanduser()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return destination
