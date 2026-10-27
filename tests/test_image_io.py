from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from verityagent_defect_detector.image_io import decode_image_bytes, load_image, resize_to_reference, validate_image_array


def test_load_valid_color_and_grayscale_images(tmp_path: Path, simple_reference: np.ndarray) -> None:
    color_path=tmp_path/"color.png"; gray_path=tmp_path/"gray.png"; assert cv2.imwrite(str(color_path),simple_reference); assert cv2.imwrite(str(gray_path),cv2.cvtColor(simple_reference,cv2.COLOR_BGR2GRAY)); assert load_image(color_path).shape==simple_reference.shape; assert load_image(gray_path).ndim==2


def test_missing_or_unsupported_images_fail(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError): load_image(tmp_path/"missing.png")
    unknown=tmp_path/"image.exe"; unknown.write_bytes(b"abc")
    with pytest.raises(ValueError,match="Unsupported"): load_image(unknown)


def test_corrupt_image_fails(tmp_path: Path) -> None:
    bad_path=tmp_path/"bad.png"; bad_path.write_bytes(b"not an image")
    with pytest.raises(ValueError,match="decode"): load_image(bad_path)


def test_array_validation_rejects_invalid_shapes_and_dtypes() -> None:
    with pytest.raises(ValueError): validate_image_array(np.zeros((0,5),dtype=np.uint8))
    with pytest.raises(ValueError,match="uint8"): validate_image_array(np.zeros((10,10),dtype=np.float32))
    with pytest.raises(ValueError,match="channels"): validate_image_array(np.zeros((10,10,2),dtype=np.uint8))


def test_decode_uploaded_bytes(simple_reference: np.ndarray) -> None:
    rgb=cv2.cvtColor(simple_reference,cv2.COLOR_BGR2RGB); buffer=BytesIO(); Image.fromarray(rgb).save(buffer,format="PNG"); decoded=decode_image_bytes(buffer.getvalue()); assert decoded.shape==simple_reference.shape; assert np.mean(np.abs(decoded.astype(np.int16)-simple_reference.astype(np.int16)))<1.0


def test_resize_same_aspect_ratio_without_distortion(simple_reference: np.ndarray) -> None:
    smaller=cv2.resize(simple_reference,(120,90)); resized,metadata=resize_to_reference(simple_reference,smaller); assert resized.shape==simple_reference.shape; assert metadata["inspection_resized"] is True


def test_mismatched_aspect_ratio_is_not_silently_distorted(simple_reference: np.ndarray) -> None:
    taller=np.zeros((240,240,3),dtype=np.uint8)
    with pytest.raises(ValueError,match="aspect ratios"): resize_to_reference(simple_reference,taller)
