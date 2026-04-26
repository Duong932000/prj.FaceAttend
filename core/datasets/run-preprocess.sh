#!/usr/bin/env bash

set -e

echo "Starting liveness detection dataset preprocessing ..."

SCRIPT_PATH="liveness/preprocess.py"
REQ_PATH="liveness/requirements.txt"

# check curl version
curl --version

# check uv install or not
if ! command -v uv &> /dev/null; then
    echo "uv could not found. Installing uv ..."
    curl -Ls https://astral.sh/uv/install.sh | bash
    # add uv to PATH
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed successfully!"
fi

# run with uv
uv run --no-cache --with-requirements "$REQ_PATH" python "$SCRIPT_PATH" \

echo "Preprocessing liveness dataset completed!"
