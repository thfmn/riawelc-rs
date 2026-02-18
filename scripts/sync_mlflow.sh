#!/bin/bash
# Sync MLflow data between local and GCS.
#
# IMPORTANT: Run "upload" before the first Vertex AI training job so the GCS
# mlruns has the same experiment IDs as local. Otherwise Vertex AI creates a
# new experiment with a different ID, and "download" brings it back as a
# duplicate.
#
# Usage:
#   bash scripts/sync_mlflow.sh upload   # local -> GCS (seed before Vertex AI)
#   bash scripts/sync_mlflow.sh download # GCS -> local (import after Vertex AI)

set -euo pipefail

BUCKET="${GCS_ARTIFACTS_BUCKET:-riawelc-artifacts-europe-west3}"
LOCAL_DIR="mlruns"
GCS_DIR="gs://${BUCKET}/mlruns"

if [ $# -eq 0 ]; then
  echo "Error: direction required."
  echo "Usage: $0 {upload|download}"
  exit 1
fi

case "$1" in
  upload)
    echo "Uploading: ${LOCAL_DIR}/ -> ${GCS_DIR}/"
    gsutil -m rsync -r "${LOCAL_DIR}/" "${GCS_DIR}/"
    ;;
  download)
    echo "Downloading: ${GCS_DIR}/ -> ${LOCAL_DIR}/"
    gsutil -m rsync -r "${GCS_DIR}/" "${LOCAL_DIR}/"
    ;;
  *)
    echo "Error: unknown direction '$1'."
    echo "Usage: $0 {upload|download}"
    exit 1
    ;;
esac

echo "Done."
