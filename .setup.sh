#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "--- SETUP START ---"

# Detect package manager
if command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
elif command -v apt &> /dev/null; then
    PKG_MANAGER="apt"
else
    echo "[ERROR] Unsupported package manager (only dnf/apt)"
    exit 1
fi

echo "[INFO] Using package manager: $PKG_MANAGER"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[INFO] Installing uv ..."
    curl -Ls https://astral.sh/uv/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "[ERROR] uv installation failed. Restart shell or add ~/.local/bin to PATH"
        exit 1
    fi
fi

# Install Python 3.10 if missing
if ! command -v python3.10 &> /dev/null; then
    echo "[INFO] Installing Python 3.10 ..."
    if [ "$PKG_MANAGER" = "dnf" ]; then
        sudo dnf install -y python3.10 python3.10-devel
    else
        sudo apt update && sudo apt install -y python3.10 python3.10-venv python3.10-dev
    fi
fi

echo "[INFO] Python 3.10 ready"

# Install system deps (fix for dnf5: use 'install @group' or list packages)
echo "[INFO] Installing system deps ..."
if [ "$PKG_MANAGER" = "dnf" ]; then
    sudo dnf install -y "@development-tools" cmake gcc gcc-c++ make python3-devel python3.10-devel \
        openblas-devel lapack-devel libjpeg-turbo-devel zlib-devel libpng-devel freetype-devel \
        ninja-build protobuf-devel || sudo dnf group install -y "Development Tools" --skip-broken
else
    sudo apt update
    sudo apt install -y build-essential cmake python3-dev python3.10-dev libopenblas-dev \
        liblapack-dev libjpeg-dev zlib1g-dev libpng-dev libfreetype6-dev ninja-build protobuf-compiler
fi

echo "[INFO] System deps installed"

# Create and activate venv
echo "[INFO] Creating .venv ..."
rm -rf .venv
uv venv --python 3.10
source .venv/bin/activate

# Install deps
echo "[INFO] Installing Python deps ..."
uv pip install --upgrade pip
uv pip install --only-binary :all: -r requirements.txt || uv pip install -r requirements.txt

echo "--- SETUP COMPLETED ---"
echo "Activate with: source .venv/bin/activate"