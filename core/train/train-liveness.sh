#!/usr/bin/env bash

set -e

IMAGE_NAME="face=attendance-liveness-train"
DOCKERFILE_PATH="core/deploy/Dockerfile"

DATASET_PATH="core/datasets/liveness"
OUTPUT_PATH=".output/liveness"

if [ ! -d "$DATASET_PATH" ]; then
    echo "Dataset not found at $DATASET_PATH. Try again ..."
    exit 1
fi

mkdir -p "$OUTPUT_PATH"

# build docker image
echo "Building docker image $IMAGE_NAME from $DOCKERFILE_PATH ..."
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" .
echo "Build completed!"

# run training liveness detection model after building the image
echo "Running training liveness detection model in docker container ..."
docker run --rm --gpus all \
    -v "$DATASET_PATH:/train/dataset" \
    -v "$OUTPUT_PATH:/train/output" \
    "$IMAGE_NAME"

echo "Training completed!"

echo "Output saved at $OUTPUT_PATH"
