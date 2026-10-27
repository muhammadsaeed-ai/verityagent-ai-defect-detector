"""Evaluate an anomaly checkpoint on labeled test/good and test/defective images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from verityagent_defect_detector.image_io import SUPPORTED_SUFFIXES
from verityagent_defect_detector.inference import predict_with_checkpoint


def evaluate(checkpoint: Path, test_dir: Path, *, image_anomaly_ratio: float, device: str) -> dict[str, Any]:
    """Compute image-level confusion metrics using affected-pixel ratio."""
    if not 0.0 <= image_anomaly_ratio <= 1.0: raise ValueError("image_anomaly_ratio must be between 0 and 1")
    observations: list[tuple[int, int, float, str]] = []
    for folder_name, label in (("good", 0), ("defective", 1)):
        folder = test_dir / folder_name
        if not folder.is_dir(): raise FileNotFoundError(f"Missing labeled test folder: {folder}")
        paths = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
        if not paths: raise ValueError(f"No supported images found in labeled test folder: {folder}")
        for path in paths:
            result = predict_with_checkpoint(checkpoint, path, device=device); score = result.affected_pixel_ratio; observations.append((label, int(score >= image_anomaly_ratio), score, path.name))
    y_true=np.array([x[0] for x in observations],dtype=np.int64); y_pred=np.array([x[1] for x in observations],dtype=np.int64); tp=int(np.sum((y_true==1)&(y_pred==1))); tn=int(np.sum((y_true==0)&(y_pred==0))); fp=int(np.sum((y_true==0)&(y_pred==1))); fn=int(np.sum((y_true==1)&(y_pred==0))); precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0; f1=2*precision*recall/(precision+recall) if precision+recall else 0.0; accuracy=(tp+tn)/len(observations) if observations else 0.0
    return {"method":"image-level classification using affected_pixel_ratio threshold","checkpoint":str(checkpoint),"test_directory":str(test_dir),"image_anomaly_ratio_threshold":image_anomaly_ratio,"sample_count":len(observations),"confusion_matrix":{"true_positive":tp,"true_negative":tn,"false_positive":fp,"false_negative":fn},"metrics":{"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1},"per_image":[{"file":name,"label":"defective" if truth else "good","predicted":"defective" if predicted else "good","affected_pixel_ratio":score} for truth,predicted,score,name in observations],"warning":"Metrics describe this labeled test set only. Choose thresholds on validation data and do not tune against the final test set."}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--test-dir",type=Path,default=Path("data/test")); parser.add_argument("--image-anomaly-ratio",type=float,default=0.005); parser.add_argument("--device",choices=("cpu","cuda"),default="cpu"); parser.add_argument("--output",type=Path,default=Path("outputs/evaluation.json")); parser.add_argument("--overwrite",action="store_true"); args=parser.parse_args(); report=evaluate(args.checkpoint,args.test_dir,image_anomaly_ratio=args.image_anomaly_ratio,device=args.device)
    if args.output.exists() and not args.overwrite: raise FileExistsError(f"Evaluation report exists: {args.output}; use --overwrite to replace it")
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8"); print(json.dumps(report,indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
