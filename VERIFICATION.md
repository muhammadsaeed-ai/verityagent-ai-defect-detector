# Verification record

## Local verification

- `pytest -q` — **31 passed**.
- `python -m compileall -q src scripts app.py` — **passed**.
- Synthetic OpenCV demo — **passed**; the included run produced one retained candidate region and the visual report assets used by the Project 09 poster.
- PyTorch smoke test — **passed**; a temporary one-epoch model was trained, checkpointed, reloaded and used for inference during local verification.

## Visual evidence

The README poster is a professional HTML/CSS/vector presentation of the executable synthetic inspection pipeline. Its inspection panel and telemetry are derived from the generated `report.json` values rather than fabricated benchmark claims.

## Environment notes

Ruff and the Streamlit dashboard were not treated as locally verified if their optional packages were unavailable in the execution environment. GitHub Actions is configured to run tests, Ruff and Python compilation for Python 3.11–3.13.

## Interpretation

The synthetic run validates implementation and localization plumbing. It does **not** establish industrial defect-detection accuracy. Real deployment requires representative labeled data, controlled imaging, calibration, validation, monitoring and human review.
