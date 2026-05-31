#!/usr/bin/env bash

set -e
cd "$(dirname "$0")"
ROOT_DIR=$(git rev-parse --show-toplevel)

echo "Activting virtual environment..."
source "$ROOT_DIR/.venv/bin/activate"

echo "Starting training face mask detection model..."
python -m core.face_mask_detection.train.training # module mode running
