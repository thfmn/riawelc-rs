#!/usr/bin/env bash
#
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
#
# Deploy RIAWELC API to Google Cloud Run.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - Docker installed and configured for Artifact Registry
#   - A .env file with RIAWELC_* variables in the project root
#   - Artifact Registry repository created in the target project
#
# Usage:
#   ./deploy/cloudrun.sh --project my-gcp-project --region europe-west3 --service-name riawelc-api
#   ./deploy/cloudrun.sh --help

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
PROJECT=""
REGION="europe-west3"
SERVICE_NAME="riawelc-api"
IMAGE_TAG="latest"
REPO_NAME="riawelc"
ENV_FILE=".env"

# --------------------------------------------------------------------------- #
# Usage / help
# --------------------------------------------------------------------------- #
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Deploy the RIAWELC API Docker image to Google Cloud Run.

Required:
  --project         GCP project ID

Optional:
  --region          GCP region             (default: europe-west3)
  --service-name    Cloud Run service name (default: riawelc-api)
  --image-tag       Docker image tag       (default: latest)
  --repo-name       Artifact Registry repo (default: riawelc)
  --env-file        Path to .env file      (default: .env)
  --help            Show this help message

Examples:
  $(basename "$0") --project my-project
  $(basename "$0") --project my-project --region us-central1 --service-name riawelc-staging
EOF
    exit 0
}

# --------------------------------------------------------------------------- #
# Parse arguments
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)
            PROJECT="$2"; shift 2 ;;
        --region)
            REGION="$2"; shift 2 ;;
        --service-name)
            SERVICE_NAME="$2"; shift 2 ;;
        --image-tag)
            IMAGE_TAG="$2"; shift 2 ;;
        --repo-name)
            REPO_NAME="$2"; shift 2 ;;
        --env-file)
            ENV_FILE="$2"; shift 2 ;;
        --help|-h)
            usage ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage ;;
    esac
done

# --------------------------------------------------------------------------- #
# Validate required arguments
# --------------------------------------------------------------------------- #
if [[ -z "${PROJECT}" ]]; then
    echo "ERROR: --project is required." >&2
    echo "Run '$(basename "$0") --help' for usage." >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Derived variables
# --------------------------------------------------------------------------- #
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPO_NAME}/${SERVICE_NAME}:${IMAGE_TAG}"

echo "============================================================"
echo "RIAWELC Cloud Run Deployment"
echo "============================================================"
echo "Project:       ${PROJECT}"
echo "Region:        ${REGION}"
echo "Service:       ${SERVICE_NAME}"
echo "Image:         ${IMAGE_URI}"
echo "Env file:      ${ENV_FILE}"
echo "============================================================"

# --------------------------------------------------------------------------- #
# Step 1: Configure Docker for Artifact Registry
# --------------------------------------------------------------------------- #
echo ""
echo "[1/4] Configuring Docker for Artifact Registry..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# --------------------------------------------------------------------------- #
# Step 2: Build Docker image
# --------------------------------------------------------------------------- #
echo ""
echo "[2/4] Building Docker image..."
docker build -t "${IMAGE_URI}" .

# --------------------------------------------------------------------------- #
# Step 3: Push to Artifact Registry
# --------------------------------------------------------------------------- #
echo ""
echo "[3/4] Pushing image to Artifact Registry..."
docker push "${IMAGE_URI}"

# --------------------------------------------------------------------------- #
# Step 4: Collect RIAWELC_* env vars from .env file
# --------------------------------------------------------------------------- #
echo ""
echo "[4/4] Deploying to Cloud Run..."

ENV_VARS=""
if [[ -f "${ENV_FILE}" ]]; then
    # Extract RIAWELC_* and OTEL_* variables, skip comments and blank lines
    while IFS='=' read -r key value; do
        # Skip comments and blank lines
        [[ -z "${key}" || "${key}" =~ ^# ]] && continue
        # Only include RIAWELC_* and OTEL_* variables
        if [[ "${key}" =~ ^RIAWELC_ || "${key}" =~ ^OTEL_ ]]; then
            if [[ -n "${ENV_VARS}" ]]; then
                ENV_VARS="${ENV_VARS},${key}=${value}"
            else
                ENV_VARS="${key}=${value}"
            fi
        fi
    done < "${ENV_FILE}"
else
    echo "WARNING: Env file '${ENV_FILE}' not found. Deploying without env vars." >&2
fi

# --------------------------------------------------------------------------- #
# Step 5: Deploy to Cloud Run
# --------------------------------------------------------------------------- #
DEPLOY_ARGS=(
    "--image=${IMAGE_URI}"
    "--platform=managed"
    "--region=${REGION}"
    "--project=${PROJECT}"
    "--memory=4Gi"
    "--cpu=2"
    "--min-instances=0"
    "--max-instances=3"
    "--port=8000"
    "--no-allow-unauthenticated"
    "--startup-cpu-boost"
)

# Add env vars if collected
if [[ -n "${ENV_VARS}" ]]; then
    DEPLOY_ARGS+=("--set-env-vars=${ENV_VARS}")
fi

# Health check / probes: Cloud Run uses startup and liveness HTTP probes
# configured via the --startup-probe and --liveness-probe flags (gcloud >= 460)
DEPLOY_ARGS+=(
    "--startup-probe-path=/health"
    "--startup-probe-period=10"
    "--startup-probe-timeout=5"
    "--startup-probe-failure-threshold=6"
    "--liveness-probe-path=/health"
    "--liveness-probe-period=30"
    "--liveness-probe-timeout=5"
    "--liveness-probe-failure-threshold=3"
)

echo ""
echo "Running: gcloud run deploy ${SERVICE_NAME} ${DEPLOY_ARGS[*]}"
echo ""

gcloud run deploy "${SERVICE_NAME}" "${DEPLOY_ARGS[@]}"

# --------------------------------------------------------------------------- #
# Print service URL
# --------------------------------------------------------------------------- #
echo ""
echo "============================================================"
echo "Deployment complete!"
echo "============================================================"

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT}" \
    --format="value(status.url)")

echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test health endpoint:"
echo "  curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" ${SERVICE_URL}/health"
echo ""
echo "NOTE: This service requires IAM authentication (--no-allow-unauthenticated)."
echo "Grant 'roles/run.invoker' to users or service accounts that need access."
echo "============================================================"
