from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from verityagent_defect_detector.alignment import align_to_reference
from verityagent_defect_detector.comparison import compare_images
from verityagent_defect_detector.config import ComparisonConfig
from verityagent_defect_detector.reporting import save_comparison_output


def test_detected_region_localizes_around_added_scratch(scratch_pair: tuple[np.ndarray, np.ndarray]) -> None:
    reference, inspection = scratch_pair
    result = compare_images(reference, inspection, config=ComparisonConfig(threshold=20, blur_kernel=1, min_area=8))
    assert result.regions
    region = max(result.regions, key=lambda item: item.area_pixels)
    assert 105 <= region.x <= 130; assert 38 <= region.y <= 60; assert region.width < 45; assert region.height > 45
    assert result.affected_pixel_ratio > 0; assert result.anomaly_score > 0


def test_clean_identical_images_have_no_candidate_regions(simple_reference: np.ndarray) -> None:
    result = compare_images(simple_reference, simple_reference.copy(), config=ComparisonConfig(min_area=5))
    assert result.regions == []; assert result.affected_pixel_ratio == 0.0; assert result.anomaly_score == 0.0; assert np.count_nonzero(result.binary_mask) == 0


def test_minimum_area_filters_tiny_noise(simple_reference: np.ndarray) -> None:
    changed = simple_reference.copy(); changed[10, 10] = (255, 255, 255)
    result = compare_images(simple_reference, changed, config=ComparisonConfig(threshold=10, blur_kernel=1, min_area=6))
    assert result.regions == []


def test_output_arrays_keep_source_dimensions(scratch_pair: tuple[np.ndarray, np.ndarray]) -> None:
    reference, inspection = scratch_pair; result = compare_images(reference, inspection, threshold=20, blur_kernel=1, min_area=8); expected = reference.shape[:2]
    assert result.inspection_image.shape[:2] == expected; assert result.annotated_image.shape[:2] == expected; assert result.heatmap.shape[:2] == expected; assert result.overlay.shape[:2] == expected; assert result.difference_map.shape == expected; assert result.binary_mask.shape == expected


def test_alignment_returns_shape_and_status(simple_reference: np.ndarray) -> None:
    shifted = cv2.warpAffine(simple_reference, np.float32([[1, 0, 2], [0, 1, -2]]), (240, 180)); aligned, metadata = align_to_reference(simple_reference, shifted, iterations=30)
    assert aligned.shape == simple_reference.shape; assert "alignment_success" in metadata


def test_report_is_json_serializable_and_output_export_is_real(tmp_path: Path, scratch_pair: tuple[np.ndarray, np.ndarray]) -> None:
    reference, inspection = scratch_pair; result = compare_images(reference, inspection, threshold=20, blur_kernel=1, min_area=8); json.dumps(result.to_report(), allow_nan=False); output_dir = tmp_path / "artifacts"; saved = save_comparison_output(output_dir, result); assert (output_dir / "report.json").exists(); report = json.loads((output_dir / "report.json").read_text(encoding="utf-8")); assert report["detected_region_count"] == len(result.regions); assert (output_dir / saved["annotated_inspection"]).stat().st_size > 0; assert len(saved) == 8


def test_export_refuses_to_overwrite_nonempty_directory(tmp_path: Path, simple_reference: np.ndarray) -> None:
    output_dir = tmp_path / "used"; output_dir.mkdir(); (output_dir / "keep.txt").write_text("keep", encoding="utf-8"); result = compare_images(simple_reference, simple_reference.copy())
    with pytest.raises(FileExistsError): save_comparison_output(output_dir, result)
    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_invalid_comparison_configuration_is_rejected() -> None:
    with pytest.raises(ValueError): ComparisonConfig(threshold=300)
    with pytest.raises(ValueError): ComparisonConfig(blur_kernel=2)
    with pytest.raises(ValueError): ComparisonConfig(mode="probability")


def test_global_brightness_shift_is_compensated(simple_reference: np.ndarray) -> None:
    brighter = np.clip(simple_reference.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    normalized = compare_images(simple_reference, brighter, config=ComparisonConfig(threshold=10, blur_kernel=1, min_area=5, normalize_illumination=True))
    unnormalized = compare_images(simple_reference, brighter, config=ComparisonConfig(threshold=10, blur_kernel=1, min_area=5, normalize_illumination=False))
    assert normalized.anomaly_score < unnormalized.anomaly_score; assert normalized.metadata["illumination_normalization_enabled"] is True; assert normalized.metadata["estimated_global_illumination_offset_reference_minus_inspection"] == -12.0


def test_grayscale_inputs_and_color_comparison_modes(scratch_pair: tuple[np.ndarray, np.ndarray]) -> None:
    reference, inspection = scratch_pair; gray_result = compare_images(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY), config=ComparisonConfig(threshold=15, blur_kernel=1, min_area=8)); color_result = compare_images(reference, inspection, config=ComparisonConfig(threshold=15, blur_kernel=1, min_area=8, mode="color")); assert gray_result.regions; assert color_result.regions; assert color_result.metadata["mode"] == "color"
