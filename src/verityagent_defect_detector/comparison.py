"""OpenCV reference-image comparison pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .alignment import align_to_reference
from .config import ComparisonConfig
from .image_io import load_image, resize_to_reference, to_bgr, to_gray


@dataclass(slots=True)
class DefectRegion:
    """A detected candidate region expressed in image-pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    area_pixels: int
    mean_difference: float

    def to_dict(self) -> dict[str, int | float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height, "area_pixels": self.area_pixels, "mean_difference": round(self.mean_difference, 4)}


@dataclass(slots=True)
class ComparisonResult:
    """All computed outputs and metrics from an image-comparison run."""

    reference_image: np.ndarray
    inspection_image: np.ndarray
    difference_map: np.ndarray
    binary_mask: np.ndarray
    heatmap: np.ndarray
    overlay: np.ndarray
    annotated_image: np.ndarray
    regions: list[DefectRegion]
    anomaly_score: float
    affected_pixel_ratio: float
    processing_time_ms: float
    metadata: dict[str, Any]

    def to_report(self) -> dict[str, Any]:
        return {
            "project": "VerityAgent AI Defect Detector",
            "method": "reference_image_comparison",
            "status": "candidate_regions_detected" if self.regions else "no_candidate_regions_detected",
            "anomaly_score": round(self.anomaly_score, 6),
            "anomaly_score_definition": "mean absolute pixel difference divided by 255; not a probability",
            "affected_pixel_ratio": round(self.affected_pixel_ratio, 6),
            "affected_pixel_percentage": round(self.affected_pixel_ratio * 100.0, 4),
            "detected_region_count": len(self.regions),
            "regions": [region.to_dict() for region in self.regions],
            "processing_time_ms": round(self.processing_time_ms, 3),
            "image_dimensions": {"width": int(self.inspection_image.shape[1]), "height": int(self.inspection_image.shape[0])},
            "metadata": self.metadata,
            "limitations": [
                "Pixel differences can be caused by lighting, pose, focus, or alignment changes.",
                "Anomaly score is a normalized image difference, not a calibrated probability.",
                "Candidate regions require human review and validation on representative data.",
            ],
        }


def _difference(reference: np.ndarray, inspection: np.ndarray, mode: str, blur_kernel: int, normalize_illumination: bool) -> tuple[np.ndarray, float | list[float]]:
    if mode == "grayscale":
        ref = to_gray(reference)
        ins = to_gray(inspection)
        if blur_kernel > 1:
            ref = cv2.GaussianBlur(ref, (blur_kernel, blur_kernel), 0)
            ins = cv2.GaussianBlur(ins, (blur_kernel, blur_kernel), 0)
        offset = float(np.median(ref.astype(np.float32) - ins.astype(np.float32))) if normalize_illumination else 0.0
        corrected = np.clip(ins.astype(np.float32) + offset, 0, 255).astype(np.uint8)
        return cv2.absdiff(ref, corrected), offset
    ref_bgr, ins_bgr = to_bgr(reference), to_bgr(inspection)
    if blur_kernel > 1:
        ref_bgr = cv2.GaussianBlur(ref_bgr, (blur_kernel, blur_kernel), 0)
        ins_bgr = cv2.GaussianBlur(ins_bgr, (blur_kernel, blur_kernel), 0)
    offsets = np.median(ref_bgr.astype(np.float32) - ins_bgr.astype(np.float32), axis=(0, 1)) if normalize_illumination else np.zeros(3, dtype=np.float32)
    corrected = np.clip(ins_bgr.astype(np.float32) + offsets, 0, 255).astype(np.uint8)
    channel_diff = cv2.absdiff(ref_bgr, corrected)
    return np.max(channel_diff, axis=2).astype(np.uint8), [float(value) for value in offsets]


def compare_images(reference: str | Path | np.ndarray, inspection: str | Path | np.ndarray, *, config: ComparisonConfig | None = None, threshold: int | None = None, blur_kernel: int | None = None, min_area: int | None = None, align: bool | None = None, mode: str | None = None) -> ComparisonResult:
    """Compare an inspection image to a known-good reference image."""
    started = perf_counter()
    base = config or ComparisonConfig()
    selected = ComparisonConfig(threshold=base.threshold if threshold is None else threshold, blur_kernel=base.blur_kernel if blur_kernel is None else blur_kernel, min_area=base.min_area if min_area is None else min_area, align=base.align if align is None else align, mode=base.mode if mode is None else mode, normalize_illumination=base.normalize_illumination)
    ref_image = load_image(reference, name="reference image")
    ins_image = load_image(inspection, name="inspection image")
    resized_inspection, resize_metadata = resize_to_reference(ref_image, ins_image)
    alignment_metadata: dict[str, Any] = {"alignment_requested": selected.align, "alignment_success": False, "alignment_method": "not requested"}
    if selected.align:
        resized_inspection, alignment_metadata = align_to_reference(ref_image, resized_inspection)
    difference_map, illumination_offset = _difference(ref_image, resized_inspection, selected.mode, selected.blur_kernel, selected.normalize_illumination)
    _, raw_mask = cv2.threshold(difference_map, selected.threshold, 255, cv2.THRESH_BINARY)
    if selected.blur_kernel >= 3:
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(raw_mask)
    regions: list[DefectRegion] = []
    inspection_bgr = to_bgr(resized_inspection)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < selected.min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
        roi = difference_map[y : y + height, x : x + width]
        regions.append(DefectRegion(x=int(x), y=int(y), width=int(width), height=int(height), area_pixels=int(round(area)), mean_difference=float(roi.mean()) if roi.size else 0.0))
    regions.sort(key=lambda item: item.area_pixels, reverse=True)
    annotated = inspection_bgr.copy()
    for index, region in enumerate(regions, start=1):
        p1 = (region.x, region.y); p2 = (region.x + region.width, region.y + region.height)
        cv2.rectangle(annotated, p1, p2, (40, 40, 245), 2)
        cv2.putText(annotated, f"Region {index}", (region.x, max(18, region.y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 245), 2, cv2.LINE_AA)
    heatmap = cv2.applyColorMap(difference_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(inspection_bgr, 0.62, heatmap, 0.38, 0)
    pixel_count = int(filtered_mask.size)
    affected_ratio = float(np.count_nonzero(filtered_mask) / pixel_count) if pixel_count else 0.0
    anomaly_score = float(np.mean(difference_map, dtype=np.float64) / 255.0)
    processing_time_ms = (perf_counter() - started) * 1000.0
    metadata = {**resize_metadata, **alignment_metadata, "mode": selected.mode, "illumination_normalization_enabled": selected.normalize_illumination, "estimated_global_illumination_offset_reference_minus_inspection": illumination_offset, "threshold_intensity_0_255": selected.threshold, "blur_kernel": selected.blur_kernel, "minimum_region_area_pixels": selected.min_area, "raw_thresholded_pixel_count": int(np.count_nonzero(raw_mask)), "retained_affected_pixel_count": int(np.count_nonzero(filtered_mask)), "score_definition": "mean absolute pixel difference / 255"}
    return ComparisonResult(reference_image=to_bgr(ref_image), inspection_image=inspection_bgr, difference_map=difference_map, binary_mask=filtered_mask, heatmap=heatmap, overlay=overlay, annotated_image=annotated, regions=regions, anomaly_score=anomaly_score, affected_pixel_ratio=affected_ratio, processing_time_ms=processing_time_ms, metadata=metadata)
