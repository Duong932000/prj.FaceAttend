#!/usr/bin/env bash

set -e
cd "$(dirname "$0")"
ROOT_DIR=$(git rev-parse --show-toplevel)

echo "Activting virtual environment..."
source "$ROOT_DIR/.venv/bin/activate"

echo "[TEST] Starting sanity inference for face mask detection..."
python -m core.face_mask_detection.inference.sanity_inference # module mode running
