from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from verityagent_defect_detector.anomaly_model import ConvAutoencoder, load_model, preprocess_for_model, reconstruction_error, save_checkpoint
from verityagent_defect_detector.config import ModelConfig
from verityagent_defect_detector.inference import predict_anomalies


def test_autoencoder_preserves_input_shape() -> None:
    model=ConvAutoencoder(image_size=32,latent_channels=4); model.eval(); inputs=torch.rand(2,3,32,32)
    with torch.inference_mode(): output=model(inputs)
    assert output.shape==inputs.shape; assert float(output.min())>=0.0; assert float(output.max())<=1.0


def test_reconstruction_error_shape_and_validation() -> None:
    source=torch.zeros(1,3,32,32); prediction=torch.ones_like(source); error=reconstruction_error(source,prediction); assert error.shape==(1,32,32); assert torch.allclose(error,torch.ones_like(error))
    with pytest.raises(ValueError): reconstruction_error(source,torch.ones(1,1,32,32))


def test_preprocessing_preserves_aspect_ratio_and_maps_back(simple_reference: np.ndarray) -> None:
    tensor,transform=preprocess_for_model(simple_reference,64); assert tensor.shape==(1,3,64,64); assert transform["resized_width"]==64; assert transform["resized_height"]==48


def test_checkpoint_round_trip_and_inference(tmp_path: Path,simple_reference: np.ndarray) -> None:
    model=ConvAutoencoder(image_size=32,latent_channels=4); model.eval(); checkpoint=tmp_path/"model.pt"; save_checkpoint(checkpoint,model,threshold=0.02,config=ModelConfig(32,4),metadata={"test":True}); loaded,metadata=load_model(checkpoint); assert isinstance(loaded,ConvAutoencoder); assert metadata["threshold"]==0.02; result=predict_anomalies(loaded,simple_reference,threshold=0.02,min_area=5); assert result.anomaly_map.shape==simple_reference.shape[:2]; assert result.heatmap.shape==simple_reference.shape; assert result.annotated_image.shape==simple_reference.shape; assert result.to_report()["method"]=="pytorch_reconstruction_anomaly_detection"


def test_checkpoint_does_not_overwrite_by_default(tmp_path: Path) -> None:
    checkpoint=tmp_path/"model.pt"; model=ConvAutoencoder(image_size=32,latent_channels=4); save_checkpoint(checkpoint,model,threshold=0.1,config=ModelConfig(32,4))
    with pytest.raises(FileExistsError): save_checkpoint(checkpoint,model,threshold=0.1,config=ModelConfig(32,4))


def test_missing_and_corrupt_checkpoints_fail(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError): load_model(tmp_path/"missing.pt")
    corrupt=tmp_path/"broken.pt"; corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError,match="safely decoded"): load_model(corrupt)


def test_invalid_checkpoint_weights_fail(tmp_path: Path) -> None:
    model=ConvAutoencoder(image_size=32,latent_channels=4); checkpoint=tmp_path/"invalid.pt"; torch.save({"format_version":1,"model_config":{"image_size":32,"latent_channels":4},"model_state_dict":{"bad":torch.ones(2)},"threshold":0.1},checkpoint)
    with pytest.raises(ValueError,match="configuration or weight"): load_model(checkpoint)
