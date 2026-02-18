#!/bin/bash

#  Copyright (C) 2026 by Tobias Hoffmann
#  thoffmann-ml@proton.me
#
#  Licensed under the MIT License.
#  For details: https://opensource.org/licenses/MIT
#
#  Author:    Tobias Hoffmann
#  Email:     thoffmann-ml@proton.me
#  License:   MIT
#  Date:      2025-2026
#  Package:   RIAWELC — Welding Defect Classification & Segmentation Pipeline

# Pre-flight check: build and test the Vertex AI container locally.
# Catches missing files, import errors, config issues, and Keras mismatches
# before spending ~12 min + cost on a real Vertex AI VM.
#
# Usage:
#   bash scripts/preflight_vertex.sh
#   bash scripts/preflight_vertex.sh --config configs/efficientnetb0_ht_1.yaml

set -euo pipefail

IMAGE="riawelc-training:preflight"
CONFIG="${1:---config configs/efficientnetb0_baseline.yaml}"

echo "==> Building container locally..."
docker build -f Dockerfile.vertex -t "${IMAGE}" .

echo "==> Running dry-run inside container..."
docker run --rm \
    -v "$(pwd)/Dataset_partitioned:/app/Dataset_partitioned:ro" \
    "${IMAGE}" \
    python scripts/01_train_classifier.py \
        --model efficientnetb0 \
        ${CONFIG} \
        --dry-run

echo "==> Pre-flight passed."
