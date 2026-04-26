#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Starting real time inference for face mask detection..."
python -m inference.realtime_inference