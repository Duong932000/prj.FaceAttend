#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Test and compare onxx and pth model after exporting ..."
python -m core.face_mask_detection.export.test_export_onnx # module mode running
