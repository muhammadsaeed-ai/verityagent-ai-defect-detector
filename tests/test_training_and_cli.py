from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from verityagent_defect_detector.cli import build_parser, main
from verityagent_defect_detector.training import NormalImageDataset, TrainingConfig, validate_dataset_separation


def _write_image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); image=np.full((48,64,3),color,dtype=np.uint8); assert cv2.imwrite(str(path),image)


def test_dataset_requires_images_and_loads_normal_images(tmp_path: Path) -> None:
    empty=tmp_path/"empty"; empty.mkdir()
    with pytest.raises(ValueError,match="No supported images"): NormalImageDataset(empty,32)
    good=tmp_path/"train"/"good"; _write_image(good/"one.png",90); dataset=NormalImageDataset(good,32); assert dataset[0].shape==(3,32,32)


def test_dataset_rejects_defective_subfolder_leakage(tmp_path: Path) -> None:
    root=tmp_path/"train"/"good"; _write_image(root/"normal.png",90); _write_image(root/"defective"/"bad.png",10)
    with pytest.raises(ValueError,match="defective/anomalous"): NormalImageDataset(root,32)


def test_train_and_validation_must_not_overlap(tmp_path: Path) -> None:
    root=tmp_path/"dataset"; (root/"good").mkdir(parents=True)
    with pytest.raises(ValueError,match="separate"): validate_dataset_separation(root,root/"good")


def test_training_config_validation() -> None:
    with pytest.raises(ValueError): TrainingConfig(epochs=0)
    with pytest.raises(ValueError): TrainingConfig(image_size=30)
    with pytest.raises(ValueError): TrainingConfig(threshold_percentile=100)


def test_cli_parser_and_required_arguments() -> None:
    parser=build_parser(); parsed=parser.parse_args(["compare","--reference","r.png","--inspection","i.png","--output-dir","out"]); assert parsed.command=="compare"
    with pytest.raises(SystemExit): parser.parse_args(["compare"])


def test_cli_compare_command_writes_real_outputs(tmp_path: Path,scratch_pair: tuple[np.ndarray,np.ndarray]) -> None:
    reference,inspection=scratch_pair; ref_path=tmp_path/"reference.png"; ins_path=tmp_path/"inspection.png"; assert cv2.imwrite(str(ref_path),reference); assert cv2.imwrite(str(ins_path),inspection); output=tmp_path/"cli-results"; code=main(["compare","--reference",str(ref_path),"--inspection",str(ins_path),"--output-dir",str(output),"--threshold","20","--blur-kernel","1","--min-area","8"]); assert code==0; assert (output/"report.json").is_file(); assert (output/"annotated_inspection.png").is_file()


def test_one_epoch_training_creates_loadable_checkpoint(tmp_path: Path) -> None:
    from verityagent_defect_detector.anomaly_model import load_model
    from verityagent_defect_detector.training import train_autoencoder
    train_dir=tmp_path/"data"/"train"/"good"; validation_dir=tmp_path/"data"/"validation"/"good"
    for index,intensity in enumerate((80,84)): _write_image(train_dir/f"train_{index}.png",intensity)
    for index,intensity in enumerate((82,86)): _write_image(validation_dir/f"validation_{index}.png",intensity)
    checkpoint=tmp_path/"models"/"model.pt"; result=train_autoencoder(train_dir,validation_dir,checkpoint,config=TrainingConfig(image_size=32,latent_channels=4,epochs=1,batch_size=2,seed=7)); assert checkpoint.is_file(); assert Path(result["history_path"]).is_file(); assert result["train_image_count"]==2; assert result["validation_image_count"]==2; model,metadata=load_model(checkpoint); assert model.image_size==32; assert 0<=metadata["threshold"]<=1
