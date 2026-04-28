#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "Creating virtual environment..."
uv venv .venv

echo "Activating..."
source .venv/bin/activate

echo "Installing dependencies (1-time)..."
uv pip install -r requirements.txt

echo "Setup completed."
