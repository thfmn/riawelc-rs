# RIAWELC Demos

Two demo interfaces for the welding defect classification and segmentation pipeline.

## Prerequisites

| Component | Required for |
|---|---|
| Python 3.11+ | Both demos |
| [uv](https://docs.astral.sh/uv/) | Both demos |
| Node.js 18+ / npm | Tauri desktop app |
| [Rust toolchain](https://rustup.rs/) | Tauri desktop app |

Install Python dependencies (from the repository root):

```bash
uv sync --group all
```

## Model Setup

### Automatic (Git LFS)

If you cloned the repository with Git LFS installed, the two model files in
`models/` are already downloaded:

```
models/
  classifier_efficientnetb0_v1.keras   (45 MB)
  segmentation_unet_v2.keras           (99 MB)
```

Verify with:

```bash
git lfs ls-files
```

If the files are LFS pointer stubs (< 1 KB each), pull them:

```bash
git lfs pull
```

### Manual (download script)

If Git LFS is not available, use the download script:

```bash
python scripts/download_models.py
```

Use `--force` to re-download existing files.

## Gradio Demo

A browser-based UI for classification, Grad-CAM visualisation, and segmentation overlay.

```bash
uv run demo/gradio_app.py
```

Opens at `http://localhost:7860`. Optional flags:

```
--model-path PATH       Classification model (default: models/classifier_efficientnetb0_v1.keras)
--seg-model-path PATH   Segmentation model   (default: models/segmentation_unet_v2.keras)
--port PORT             Server port           (default: 7860)
--share                 Create a public Gradio link
```

## Tauri Desktop App

A native desktop app built with Tauri 2 + SvelteKit that talks to the
FastAPI backend.

### 1. Start the API server

```bash
uv run -m uvicorn riawelc.api.main:app --reload
```

The API runs at `http://localhost:8000`.

### 2. Install frontend dependencies and launch

```bash
cd demo/tauri-app
npm install
npm run tauri dev
```

This compiles the Rust backend, starts the Vite dev server, and opens the
native window.
