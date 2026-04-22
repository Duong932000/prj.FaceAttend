FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /train

RUN apt-get update && apt-get install -y curl \
    && curl -Ls https://astral.sh/uv/install.sh | bash \
    && rm -rf /var/lib/apt/lists/*

# copy code
COPY . .

# install dependencies (correct path)
RUN uv pip install --system --no-cache \
    -r core/train/training-loops/liveness/requirements.txt

# run training + export
CMD ["sh", "-c", "python core/train/training-loops/liveness/liveness-training.py && python core/train/export/liveness/export_onnx.py"]

