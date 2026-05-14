#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

ROOT_DIR=$(git rev-parse --show-toplevel)

echo "Activating virtual environment..."
source "$ROOT_DIR/.venv/bin/activate"

export FACE_ATTEND_ROOT="$ROOT_DIR"

echo "FACE_ATTEND_ROOT=$FACE_ATTEND_ROOT"

echo "[INFO] Running auto_capture_dataset from webcam..."
python -m core.face_recognition.datasets.auto_capture_dataset # module mode