#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "--- UPDATE VENV ---"

# check .venv exist or not
if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv not found. Run ./.setup.sh first !"
    exit 1
fi

# activate .venv
source .venv/bin/activate

# sync requirements.txt
echo "[INFO] Syncing requirements.txt with uv pip sync ..."
uv pip sync requirements.txt

echo "--- UPDATE VENV COMPLETED ---"
echo "All packages from requirements.txt are now installed."
echo "Activate venv: $(which python)"
