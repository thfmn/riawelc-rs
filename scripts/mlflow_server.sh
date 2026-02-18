#!/bin/bash
# Launch MLflow tracking UI
#
# Usage:
#   bash scripts/mlflow_server.sh
#   bash scripts/mlflow_server.sh mlruns 5001

BACKEND_STORE="${1:-mlruns}"
PORT="${2:-5000}"

echo "Starting MLflow UI at http://localhost:${PORT}"
mlflow ui --backend-store-uri "${BACKEND_STORE}" --port "${PORT}"
