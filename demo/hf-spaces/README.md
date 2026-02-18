---
title: RIAWELC — Welding Defect Inspector
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
python_version: "3.11"
license: mit
pinned: false
tags:
  - welding
  - defect-detection
  - classification
  - segmentation
  - grad-cam
  - radiography
  - ndt
---

# RIAWELC — Welding Defect Inspector

Upload a radiographic weld image for defect classification, Grad-CAM
localization, and U-Net segmentation overlay.

## Models

| Task             | Architecture    | Input           |
|------------------|-----------------|-----------------|
| Classification   | EfficientNetB0  | 227x227 gray    |
| Segmentation     | U-Net (EffB0)   | 224x224 gray    |

**Classes**: crack, lack_of_penetration, no_defect, porosity

## Dataset

Trained on the RIAWELC-RS dataset (21,964 radiographic weld images).

## Deployment to Hugging Face Spaces

1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space):
   - SDK: **Gradio**
   - Hardware: **CPU Basic** (or upgrade for faster inference)

2. Upload the contents of this directory (`demo/hf-spaces/`) to the Space
   repository root.

3. Upload trained model weights to the Space:
   - `models/classifier/best.keras` (EfficientNetB0 classifier)
   - `models/segmentation/best.keras` (U-Net segmentation, optional)

   Alternatively, set the `MODEL_PATH` and `SEG_MODEL_PATH` Space
   variables to point to a Hugging Face model repository.

4. The Space will automatically install dependencies from
   `requirements.txt` and launch `app.py`.

## Local Testing

```bash
cd demo/hf-spaces
pip install -r requirements.txt
python app.py
```

## License

MIT -- See the [RIAWELC repository](https://github.com/thoffmann-ml/RIAWELC)
for full details.
