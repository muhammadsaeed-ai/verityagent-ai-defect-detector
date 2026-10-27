# Build verification record

Local verification performed during repository assembly on 2026-10-27:

- `pytest -q`: **31 passed**.
- `python -m compileall -q src scripts app.py`: passed.
- Package editable-build/install check: passed with the environment's installed build backend (`--no-build-isolation`, because this sandbox cannot reach the package index).
- `verity-defect --help`: passed; both `compare` and `infer` commands are registered.
- OpenCV synthetic scratch run: produced one candidate region and exported the difference map, binary mask, heatmap, overlay, annotated image, and JSON report. The JSON values and preview image are included under `examples/results/`.
- PyTorch smoke workflow: trained a one-epoch model on a tiny temporary defect-free sample, safely reloaded the checkpoint, ran inference, and generated an evaluation report. Those temporary smoke-test files are not bundled as a model or presented as accuracy evidence.

Environment limitations:

- Ruff could not be installed or run in the authoring sandbox because the `ruff` executable was absent and outbound package-index access was unavailable. The repository includes `ruff check .` in its GitHub Actions workflow for execution in CI.
- The Streamlit dashboard source compiled successfully, but the dashboard itself was not launched because Streamlit is not installed in this sandbox. Install the `app` extra and run `streamlit run app.py` to launch it.

The synthetic demo validates program flow and scratch localization only. It is not evidence of real-world industrial defect-detection accuracy.
