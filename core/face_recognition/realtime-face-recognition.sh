#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"

echo "Activting virtual environment..."
source .venv/bin/activate

echo "Real-time inference face recognition ..."
python -m inference.realtime
