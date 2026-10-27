# Troubleshooting

### OpenCV cannot decode an image

Use PNG/JPG/JPEG/BMP/TIF/TIFF/WEBP and verify the file opens locally. Files above 25 MB and decoded images above 40 million pixels are rejected.

### Aspect ratios differ

Crop both images to the same field of view or capture them with the same framing. The tool refuses material aspect-ratio differences instead of silently stretching or cropping product details.

### Lighting changes are highlighted

Use consistent illumination and exposure. Try a small Gaussian blur, adjust the intensity threshold, enable ECC alignment for small shifts, and inspect the difference map.

### No candidate regions

Check alignment, lower the pixel threshold, and check minimum region area. Inspect the difference map before concluding that the object is defect-free.

### Trained-model inference is unavailable

The repository does not ship a fictional pre-trained checkpoint. Prepare defect-free training and validation images and run `scripts/train_model.py`, or use reference comparison without a checkpoint.

### Autoencoder uses too much memory

Reduce batch size or use `--image-size 128`. CPU training is supported; CUDA is optional.

### Streamlit is not installed

Run `pip install -e ".[app]"`, then `streamlit run app.py`.

### Output directory already contains files

Exports refuse to overwrite non-empty directories. Use a new output directory or move the previous artifacts.
