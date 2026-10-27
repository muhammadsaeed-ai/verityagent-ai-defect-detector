# Dataset preparation

The trained model learns **normal/defect-free** product appearance. Training and threshold calibration must not contain labeled defective samples.

```text
data/
  train/good/
  validation/good/
  test/good/
  test/defective/
```

## Collection guidance

- Capture the same product type, camera, field of view, lighting and focus intended for inference.
- Include normal variation in position, finish and acceptable manufacturing tolerance.
- Keep images from the same physical unit or capture session in one split to reduce leakage.
- Keep validation defect-free because its error distribution determines the model threshold.
- Reserve the final labeled test split for evaluation; do not tune against it repeatedly.
- Check licensing and commercial-use terms for every external dataset.
- Keep preprocessing consistent: aspect-preserving letterbox to RGB.

## Evaluation

Run `python scripts/evaluate_model.py --checkpoint models/defect_autoencoder.pt --test-dir data/test`. An image is predicted defective when the fraction of pixels above the learned pixel threshold is at least `--image-anomaly-ratio` (default `0.005`). Choose this image-level decision threshold using validation data.

The evaluation script reports a confusion matrix and image-level accuracy, precision, recall and F1 for the supplied labeled samples. These metrics describe only that dataset and protocol.
