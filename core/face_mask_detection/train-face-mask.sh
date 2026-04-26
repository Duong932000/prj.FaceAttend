#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

echo "Activting virtual environment..."
source .venv/bin/activate

echo "Starting training face mask detection model..."
python -m train.facemask_training
