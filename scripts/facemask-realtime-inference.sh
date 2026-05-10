#!/usr/bin/env bash

set -e
cd "$(dirname "$0")"
ROOT_DIR=$(git rev-parse --show-toplevel)

echo "Activting virtual environment..."
source "$ROOT_DIR/.venv/bin/activate"

echo "[TEST] Starting inference for realtime face mask detection using Webcam..."
python -m core.face_mask_detection.inference.realtime_inference
