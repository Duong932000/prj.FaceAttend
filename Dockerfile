
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1 libgomp1 v4l-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install insightface onnxruntime-gpu opencv-python numpy

COPY recognition.py .

CMD ["python3", "recognition.py"]
