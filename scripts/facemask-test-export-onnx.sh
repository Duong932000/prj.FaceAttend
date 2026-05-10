#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Test and compare onxx and pth model after exporting ..."
python -m export.test_export_onnx
