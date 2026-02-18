#!/bin/bash
# Upload local dataset, checkpoints, and pseudo-masks to GCS
#
# Usage:
#   bash scripts/upload_to_gcs.sh data                 # upload dataset
#   bash scripts/upload_to_gcs.sh checkpoints           # upload checkpoints
#   bash scripts/upload_to_gcs.sh pseudomasks            # upload pseudo-masks
#   bash scripts/upload_to_gcs.sh all                   # upload all
#   bash scripts/upload_to_gcs.sh --dry-run data        # preview only

set -euo pipefail

# Load .env if present (provides bucket names, project ID, etc.)
if [[ -f .env ]]; then set -a; source .env; set +a; fi

DATA_BUCKET="${RIAWELC_DATA_BUCKET:?Set RIAWELC_DATA_BUCKET in .env}"
ARTIFACTS_BUCKET="${RIAWELC_ARTIFACTS_BUCKET:?Set RIAWELC_ARTIFACTS_BUCKET in .env}"
LOCAL_DATA="Dataset_partitioned"
LOCAL_CHECKPOINTS="outputs/models/checkpoints"
LOCAL_PSEUDOMASKS="outputs/pseudomasks"
DRY_RUN=false

usage() {
  echo "Usage: $0 [--dry-run] {data|checkpoints|pseudomasks|all}"
  exit 1
}

# Parse --dry-run flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    data|checkpoints|pseudomasks|all) COMMAND="$1"; shift ;;
    *) usage ;;
  esac
done

[[ -z "${COMMAND:-}" ]] && usage

verify_count() {
  local gcs_path="$1"
  local description="$2"
  local count
  count=$(gsutil ls -r "${gcs_path}/**" 2>/dev/null | grep -c '\.png$\|\.h5$\|\.keras$' || true)
  echo "  ${description}: ${count} files on GCS"
}

upload_data() {
  echo "==> Uploading dataset to gs://${DATA_BUCKET}/"
  if [[ ! -d "${LOCAL_DATA}" ]]; then
    echo "Error: ${LOCAL_DATA}/ not found" >&2
    exit 1
  fi

  for split in training validation testing; do
    local src="${LOCAL_DATA}/${split}"
    local dst="gs://${DATA_BUCKET}/${split}/"
    if [[ ! -d "${src}" ]]; then
      echo "Warning: ${src}/ not found, skipping" >&2
      continue
    fi
    local count
    count=$(find "${src}" -name '*.png' | wc -l)
    echo "  ${split}: ${count} local files"
    if [[ "${DRY_RUN}" == true ]]; then
      echo "  [dry-run] gsutil -m rsync -r ${src}/ ${dst}"
    else
      gsutil -m rsync -r "${src}/" "${dst}"
    fi
  done

  if [[ "${DRY_RUN}" == false ]]; then
    echo "==> Verifying data upload"
    for split in training validation testing; do
      verify_count "gs://${DATA_BUCKET}/${split}" "${split}"
    done
  fi
}

upload_checkpoints() {
  echo "==> Uploading checkpoints to gs://${ARTIFACTS_BUCKET}/outputs/checkpoints/"
  if [[ ! -d "${LOCAL_CHECKPOINTS}" ]]; then
    echo "Error: ${LOCAL_CHECKPOINTS}/ not found" >&2
    exit 1
  fi

  local count
  count=$(find "${LOCAL_CHECKPOINTS}" -type f | wc -l)
  echo "  ${count} local checkpoint files"

  if [[ "${DRY_RUN}" == true ]]; then
    echo "  [dry-run] gsutil -m rsync -r ${LOCAL_CHECKPOINTS}/ gs://${ARTIFACTS_BUCKET}/outputs/checkpoints/"
  else
    gsutil -m rsync -r "${LOCAL_CHECKPOINTS}/" "gs://${ARTIFACTS_BUCKET}/outputs/checkpoints/"
    echo "==> Verifying checkpoint upload"
    verify_count "gs://${ARTIFACTS_BUCKET}/outputs/checkpoints" "checkpoints"
  fi
}

upload_pseudomasks() {
  echo "==> Uploading pseudo-masks to gs://${ARTIFACTS_BUCKET}/outputs/pseudomasks/"
  if [[ ! -d "${LOCAL_PSEUDOMASKS}" ]]; then
    echo "Error: ${LOCAL_PSEUDOMASKS}/ not found" >&2
    exit 1
  fi

  for class_dir in "${LOCAL_PSEUDOMASKS}"/*/; do
    local class_name
    class_name=$(basename "${class_dir}")
    local src="${LOCAL_PSEUDOMASKS}/${class_name}"
    local dst="gs://${ARTIFACTS_BUCKET}/outputs/pseudomasks/${class_name}/"
    local count
    count=$(find "${src}" -name '*.png' | wc -l)
    echo "  ${class_name}: ${count} local files"
    if [[ "${DRY_RUN}" == true ]]; then
      echo "  [dry-run] gsutil -m rsync -r ${src}/ ${dst}"
    else
      gsutil -m rsync -r "${src}/" "${dst}"
    fi
  done

  if [[ "${DRY_RUN}" == false ]]; then
    echo "==> Verifying pseudo-mask upload"
    verify_count "gs://${ARTIFACTS_BUCKET}/outputs/pseudomasks" "pseudomasks"
  fi
}

case "${COMMAND}" in
  data)        upload_data ;;
  checkpoints) upload_checkpoints ;;
  pseudomasks) upload_pseudomasks ;;
  all)         upload_data; upload_checkpoints; upload_pseudomasks ;;
esac

echo "Done."
