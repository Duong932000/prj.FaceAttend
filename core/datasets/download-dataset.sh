#!/usr/bin/env bash

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <kaggle-dataset-name>"
    echo "Example: "
    echo "    $0 shahabhasan/face-anti-spoofing-dataset"
    exit 1
fi

DATASET=$1

BASE_DATASET_DIR="$HOME/ml-dataset/kaggle"

mkdir -p "$BASE_DATASET_DIR"
cd "$BASE_DATASET_DIR"

echo "Fast downloading dataset from: $DATASET ..."

# extract dataset name after last slash
DATASET_NAME=$(basename "$DATASET")

ZIP_FILE="${DATASET_NAME}.zip"
EXTRACT_DIR="$DATASET_NAME"

# check kaggle CLI installed or not
kaggle --version

# download dataset with kaggle CLI
kaggle datasets download -d "$DATASET"

# extract
echo "Extracting $ZIP_FILE -> $EXTRACT_DIR ..."
unzip -o "$ZIP_FILE" -d "$EXTRACT_DIR"

echo "Dataset downloaded and extracted successfully!"
echo "Dataset path: $BASE_DATASET_DIR/$EXTRACT_DIR"

