# RIAWELC-RS: Welding Radiograph Classification & Weakly-Supervised Segmentation

Classification and weakly-supervised segmentation of welding radiographs using transfer learning, Grad-CAM localization, and a simple FastAPI backend with edge model export.

```
Radiograph → Classification → Grad-CAM → Pseudo-Masks → U-Net Segmentation
```

## Dataset

This project uses **RIAWELC-RS** (Radiograph-Split), a corrected version of the original [RIAWELC](https://github.com/stefyste/RIAWELC) dataset (University of Calabria, 2023). The original dataset contains 24,407 grayscale 227×227 radiographic images across 4 defect classes, extracted from 29 high-resolution industrial radiographs. RIAWELC-RS removes 2,443 exact duplicate images and re-splits at the radiograph level to eliminate data leakage, resulting in **21,964 unique images**.

> **Note on image dimensions:** The original paper and GitHub README both state 224×224, but the actual distributed images are 227×227 pixels. This likely reflects the native input size of SqueezeNet/AlexNet (227×227) used by the original authors, misreported as the more commonly cited 224×224 (VGG/ResNet). All classifier configs in this project use 227×227 to match the true on-disk dimensions. The U-Net segmentation pipeline resizes to 224×224 because the encoder-decoder architecture requires dimensions that halve cleanly through 5 pooling stages (224→112→56→28→14→7).

### RIAWELC vs RIAWELC-RS

| | RIAWELC (original) | RIAWELC-RS (this project) |
|---|---|---|
| Total images | 24,407 | 21,964 |
| Duplicates | 2,443 across splits | Removed |
| Split strategy | Random (patch-level) | Radiograph-level |
| Data leakage | Yes | None |

### RIAWELC-RS Split

| Class | Code | Train | Val | Test |
|-------|------|-------|-----|------|
| Crack | CR | 3,043 | 521 | 442 |
| Lack of Penetration | LP | 4,684 | 1,577 | 609 |
| No Defect | ND | 4,175 | 684 | 541 |
| Porosity | PO | 3,761 | 790 | 1,137 |
| **Total** | | **15,663** | **3,572** | **2,729** |

## Data Leakage in the Original RIAWELC Dataset

### The Problem

The original RIAWELC dataset contains a severe data leakage issue. The 29 source radiographs (2000×8640 pixels each) were sliced into 24,407 patches using a sliding window with overlap, then resized to 227×227 (see note above). The original authors pooled all patches and randomly assigned them to train (65%), validation (25%), and test (10%) splits.

MD5 checksum verification reveals that **2,443 of 2,443 test images (100%) are byte-for-byte identical to images in the training set**. The entire test set is a subset of the training data. Any model evaluated on these splits is partially evaluated on its own training examples.

Beyond exact duplicates, overlapping sliding-window patches from the same physical radiograph appear across splits, leaking spatial context even where images are not identical.

### Impact on Published Results

Several papers report classification results on RIAWELC using the original splits or drawing from the same undeduplicated image pool. The table below lists published accuracies we are aware of. Because the original test set consists entirely of training duplicates, these metrics do not reflect generalization to unseen data and should be interpreted with caution.

| Paper | Year | Journal | Reported Metric | Split Method |
|---|---|---|---|---|
| [Totino et al.](https://doi.org/10.53375/ijecer.2023.320) | 2023 | IJECER | 93.33% acc. | Original 65/25/10 splits |
| [Palma-Ramírez et al.](https://doi.org/10.1016/j.heliyon.2024.e30590) | 2024 | Heliyon | 98.75% acc. | 5-fold CV on 1,600 balanced samples from same image pool |
| [Xia et al.](https://doi.org/10.1088/1361-6501/ae09ce) | 2025 | Meas. Sci. Technol. | >92% acc. | RIAWELC as intermediate fine-tuning source |
| [Ngo Thi Hoa et al.](https://doi.org/10.1177/16878132251341615) | 2025 | Adv. Mech. Eng. | 99.83% acc. | Full 24,407 images, original splits |
| [López et al.](https://doi.org/10.3390/s25196183) | 2025 | Sensors | 99.87% F1 | 10-fold CV on merged train+val (21,964), tested on original 2,443 test set |

None of these papers report investigating or addressing the duplicate issue. For comparison, the EfficientNetB0 classifier in this project achieves **81.86%** on the leakage-free RIAWELC-RS test split (see [Results](#results) below).

> **Note:** We do not suggest that the above results are without merit in other respects; architectural contributions, augmentation strategies, and transfer learning methods may well be valuable. However, the headline accuracy figures on RIAWELC cannot be taken at face value given that the evaluation data is contained in the training data.

### RIAWELC-RS: The Fix

RIAWELC-RS addresses both forms of leakage:

1. **Deduplication.** All 2,443 duplicate images are removed, reducing the dataset from 24,407 to 21,964 unique images.
2. **Radiograph-level splitting.** Patches are grouped by their source radiograph, identified from filenames (e.g., `RRT-30R_Img1_A80_S1_[9][58].png` maps to radiograph `RRT-30R`). Every patch from a given radiograph goes to exactly one split, ensuring zero information leakage between training and evaluation.

### RIAWELC-RS Final Split

| Split | Radiographs | crack | lack_of_pen. | no_defect | porosity | Total |
|---|---|---|---|---|---|---|
| Train (71.3%) | 15 | 3,043 | 4,684 | 4,175 | 3,761 | 15,663 |
| Val (16.3%) | 8 | 521 | 1,577 | 684 | 790 | 3,572 |
| Test (12.4%) | 6 | 442 | 609 | 541 | 1,137 | 2,729 |

### Class Weights

Training uses inverse-frequency class weights (`total / (n_classes * count)`) to compensate for imbalance, computed automatically from the training split at runtime.

| Class | % of Train | Weight |
|---|---|---|
| crack | 19.4% | 1.29 |
| lack_of_penetration | 29.9% | 0.84 |
| no_defect | 26.7% | 0.94 |
| porosity | 24.0% | 1.04 |

### Verify / Reproduce

```bash
# Verify current splits are clean (checksums + radiograph-level check)
python scripts/00_verify_splits.py

# Reproduce the split from scratch
python scripts/00_resplit_by_radiograph.py --dry-run  # preview
python scripts/00_resplit_by_radiograph.py             # execute
```

## Results

All metrics evaluated on the held-out test split (6 radiographs, 2,729 patches) with radiograph-level splitting to prevent data leakage.

### Classification (EfficientNetB0)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Crack | 0.47 | 0.79 | 0.59 | 442 |
| Lack of Penetration | 0.76 | 0.36 | 0.49 | 609 |
| No Defect | 0.99 | 0.98 | 0.99 | 541 |
| Porosity | 0.98 | 1.00 | 0.99 | 1,137 |
| **Overall Accuracy** | | | **81.86%** | **2,729** |

### Segmentation (U-Net, EfficientNetB0 encoder)

Evaluated on the 3 defect classes only (`no_defect` excluded, since there is no spatial defect to segment).

| Class | IoU | Dice | Samples |
|-------|-----|------|---------|
| Crack | 0.456 ± 0.218 | 0.580 ± 0.204 | 442 |
| Lack of Penetration | 0.511 ± 0.236 | 0.629 ± 0.213 | 609 |
| Porosity | 0.717 ± 0.130 | 0.772 ± 0.090 | 1,137 |
| **Overall** | **0.607 ± 0.218** | **0.693 ± 0.180** | **2,188** |

**Weakly-supervised context:** Segmentation ground truth consists of pseudo-masks derived from Grad-CAM, not human annotations. The model learns to refine noisy class activation maps into sharper contours. A mean IoU of 0.61 is consistent with published weakly-supervised segmentation results (typically 0.50–0.70) and demonstrates that the pipeline extracts useful spatial information without any manual mask labeling.

## Quick Start

```bash
# 1. Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync --group all

# 2. Verify setup
uv run pytest tests/ -q

# 3. Dry-run to check model builds
python scripts/01_train_classifier.py --model efficientnetb0 --config configs/efficientnetb0_baseline.yaml --dry-run
```

## Training Pipeline

### Step 1 — Train Classifiers

```bash
# Train EfficientNetB0 (recommended first)
python scripts/01_train_classifier.py \
    --model efficientnetb0 \
    --config configs/efficientnetb0_baseline.yaml

# Train ResNet50V2
python scripts/01_train_classifier.py \
    --model resnet50v2 \
    --config configs/resnet50v2_baseline.yaml

# Train all registered models
python scripts/01_train_classifier.py --model all

# Dry-run (builds model, prints summary, no training)
python scripts/01_train_classifier.py --model efficientnetb0 --dry-run

# Disable experiment tracking
python scripts/01_train_classifier.py \
    --model efficientnetb0 \
    --config configs/efficientnetb0_baseline.yaml \
    --tracking none
```

### Step 2 — Evaluate

```bash
python scripts/02_evaluate_classifier.py \
    --model efficientnetb0 \
    --config configs/efficientnetb0_baseline.yaml

# Or specify a checkpoint directly
python scripts/02_evaluate_classifier.py \
    --model-path outputs/models/checkpoints/efficientnetb0/v1/best.keras \
    --config configs/efficientnetb0_baseline.yaml
```

Outputs: `outputs/evaluation/confusion_matrix.png`, `classification_report.csv`, `metrics.csv`

### Step 3 — Grad-CAM Heatmaps

```bash
python scripts/03_generate_gradcam.py \
    --model-path outputs/models/checkpoints/efficientnetb0/v1/best.keras \
    --config configs/gradcam.yaml

# Limit to 20 samples per class
python scripts/03_generate_gradcam.py \
    --model-path outputs/models/checkpoints/efficientnetb0/v1/best.keras \
    --num-samples 20
```

Outputs: `outputs/gradcam/overlays/`, `outputs/gradcam/heatmaps/`

### Step 4 — Pseudo-Masks

```bash
python scripts/04_generate_pseudomasks.py

# Adjust threshold and minimum area
python scripts/04_generate_pseudomasks.py --threshold 0.4 --min-area 50
```

Outputs: `outputs/pseudomasks/`

### Step 5 — Train Segmentation

```bash
python scripts/05_train_segmentation.py --config configs/segmentation_unet.yaml

# Dry-run
python scripts/05_train_segmentation.py --config configs/segmentation_unet.yaml --dry-run
```

### Step 6 — Evaluate Segmentation

```bash
python scripts/06_evaluate_segmentation.py \
    --model-path outputs/models/checkpoints/unet_efficientnetb0/v1/best.keras
```

Outputs: `outputs/evaluation/segmentation/segmentation_grid.png`, `per_sample_metrics.csv`, `per_class_metrics.csv`, `summary_statistics.csv`, `iou_histogram.png`

### Step 7 — Export to ONNX

```bash
python scripts/07_export_model.py \
    --format onnx \
    --model-path outputs/models/checkpoints/efficientnetb0/v1/best.keras

# With validation and benchmarking
python scripts/07_export_model.py \
    --format onnx \
    --model-path outputs/models/checkpoints/efficientnetb0/v1/best.keras \
    --validate --benchmark
```

Outputs: `outputs/exported/model.onnx`

## Vertex AI (Cloud Training)

Train on GPU-equipped VMs in Google Cloud. The training script runs identically
locally and on Vertex AI. Only the `--tracking` and path flags change.

### Setup

```bash
# Install Vertex AI dependencies (separate from local dev)
uv sync --group vertex

# Build and push the training container
gcloud builds submit --config cloudbuild.yaml --region=europe-west3
```

### Submit Jobs

```bash
# Dry-run — preview job config and cost estimate
python scripts/submit_vertex_job.py --model efficientnetb0 --dry-run

# Submit a GPU training job (Spot VM)
python scripts/submit_vertex_job.py --model efficientnetb0 --yes

# Fine-tune from a checkpoint with a custom config
python scripts/submit_vertex_job.py --model efficientnetb0 \
    --config configs/efficientnetb0_ht_1.yaml \
    --resume-from gs://$GCS_ARTIFACTS_BUCKET/outputs/checkpoints/efficientnetb0/v1/feature_extraction/best.keras \
    --yes

# CPU-only smoke test
python scripts/submit_vertex_job.py \
    --model efficientnetb0 --machine-type cpu-only --yes
```

### MLflow Sync Workflow

MLflow FileStore identifies experiments by a numeric ID stored in the directory
name. Local and GCS each maintain their own `mlruns/` directory. If GCS has
never seen the local experiment IDs, Vertex AI jobs create new experiments with
different IDs, causing duplicates after download.

**Required workflow:**

```bash
# 1. BEFORE the first Vertex AI job: seed GCS with local experiments
./scripts/sync_mlflow.sh upload

# 2. Submit the Vertex AI job (runs on GCS mlruns)
python scripts/submit_vertex_job.py --model efficientnetb0 \
    --config configs/efficientnetb0_ht_1.yaml --yes

# 3. AFTER job completes: pull results back to local
./scripts/sync_mlflow.sh download

# 4. View in MLflow
bash scripts/mlflow_server.sh
```

Step 1 ensures GCS has the same experiment IDs as local. The Vertex AI job
then adds runs to the existing experiment instead of creating a duplicate.

See the Vertex AI section above for setup instructions. Full onboarding
documentation (GCP APIs, buckets, IAM, troubleshooting) is maintained internally.

## API

```bash
# Start the API server
uv run uvicorn riawelc.api.main:create_app --factory --reload

# Or with Docker
docker compose up api
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe (model loaded?) |
| `/predict` | POST | Classify image + Grad-CAM overlay |
| `/segment` | POST | Grad-CAM segmentation mask |
| `/segment/unet/baseline` | POST | U-Net segmentation (baseline) |
| `/segment/unet/augmented` | POST | U-Net segmentation (augmented) |
| `/model/info` | GET | Model metadata |

```bash
# Example: classify an image
curl -X POST http://localhost:8000/predict \
    -F "file=@Dataset_partitioned/testing/crack/sample.png"
```

## MLflow

```bash
bash scripts/mlflow_server.sh                  # local backend, http://localhost:5000
bash scripts/mlflow_server.sh gs://bucket/mlruns  # GCS backend
docker compose --profile dev up                # API + MLflow together
```

## Project Structure

```
RIAWELC/
├── src/riawelc/          # Source code
│   ├── config.py         #   TrainingConfig dataclass
│   ├── data/             #   tf.data pipelines, augmentation
│   ├── models/           #   Classifiers, Grad-CAM, segmentation, export
│   ├── training/         #   Training loop, callbacks
│   ├── inference/        #   Prediction pipeline
│   └── api/              #   FastAPI backend
├── scripts/              # CLI entrypoints (01-07) + Vertex AI submission
├── configs/              # YAML training configs
├── tests/                # pytest test suite
├── Dockerfile.vertex     # Vertex AI training container (TF 2.19.1-gpu)
├── cloudbuild.yaml       # Cloud Build → Artifact Registry
└── Dataset_partitioned/  # RIAWELC dataset (not tracked in git)
```

## Configuration

Training is configured via YAML files in `configs/`. Example:

```yaml
model:
  name: efficientnetb0
  input_shape: [227, 227, 1]
  num_classes: 4
  freeze_backbone: true
  fine_tune_at: 100
```

See `configs/efficientnetb0_baseline.yaml` for a complete example.

CLI flags can override config values for cloud training:

```bash
python scripts/01_train_classifier.py \
    --model efficientnetb0 \
    --config configs/efficientnetb0_baseline.yaml \
    --tracking both \
    --data-dir /gcs/$GCS_DATA_BUCKET \
    --checkpoint-dir /gcs/$GCS_ARTIFACTS_BUCKET/outputs/checkpoints \
    --mlflow-uri /gcs/$GCS_ARTIFACTS_BUCKET/mlruns
```

## Tech Stack

| Component | Choice |
|-----------|--------|
| ML Framework | TensorFlow/Keras (Functional API) |
| Classifiers | EfficientNetB0, ResNet50V2 |
| Segmentation | U-Net (native, EfficientNetB0 encoder) |
| Edge Export | ONNX (tf2onnx) |
| API | FastAPI + Pydantic |
| Experiment Tracking | MLflow + Vertex AI Experiments |
| Cloud Training | Vertex AI (europe-west3, T4 GPU) |
| Desktop Demo | Tauri + Svelte |
| Package Manager | uv |

## Dataset Citation

> Benito Totino, Fanny Spagnolo, Stefania Perri, "RIAWELC: A Novel Dataset of Radiographic Images for Automatic Weld Defects Classification", ICMECE 2022.

## License

MIT
