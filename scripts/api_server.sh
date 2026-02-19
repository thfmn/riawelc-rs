#!/bin/bash
# Launch the RIAWELC FastAPI backend
#
# Usage:
#   bash scripts/api_server.sh
#   bash scripts/api_server.sh 8080

PORT="${1:-8000}"

echo "Starting RIAWELC API at http://localhost:${PORT}"
uv run uvicorn riawelc.api.main:create_app --factory --host 0.0.0.0 --port "${PORT}"
