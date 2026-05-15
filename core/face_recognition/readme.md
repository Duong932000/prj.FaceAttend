
Face Recognition System

# 1. Folder contains the complete face recognition pipeline build on top of InsightFace and FAISS

The system supports:
- Auto dataset collection from webcam
- Face alignment and preprocessing
- Embedding generation using pretrained ArcFace
- FAISS vector indexing
- Realtime face recognition inference
- Local pretrained model management

# 2. Architecture Overview

`
Webcam
   ↓
InsightFace Detection
   ↓
Face Alignment
   ↓
ArcFace Embedding Extraction
   ↓
FAISS Similarity Search
   ↓
Identity Matching
`

# 3. Pretrained Workflow

This project currently uses pretrained InsightFace models, used is: "buffalo_l"
Following some step below:

- step 1: Download pretrained model:
    `core.face_recognition.download.download_model.py`

- step 2: Automatic collect dataset
    `core.face_recognition.enrollment.faceid_enrollment.py`

- step 3: Face Alignment
    `core.face_recognition.preprocess.align_face.py`

- step 4: Generate Embeddings
    `core.face_recognition.preprocess.generate_embeddings.py`

- step 5: Build FAISS index
    `core.face_recognition.preprocess.build_faiss_indexes.py`

# 3. Folder Struture

`
face_recognition
|
├── ./configs
│   ├── ./configs/common.yml
│   ├── ./configs/export_onnx.yml
│   └── ./configs/export_tensorRT.yml
├── ./download
│   ├── ./download/download_model.py
│   └── ./download/__int__.py
├── ./enrollment
│   ├── ./enrollment/faceid_enrollment.py
│   └── ./enrollment/__init__.py
├── ./export
│   ├── ./export/export_onnx.py
│   └── ./export/export_tensorRT.py
├── ./inference
│   └── ./inference/__init__.py
├── ./__init__.py
├── ./models
│   ├── ./models/backbones
│   │   ├── ./models/backbones/iresnet.py
│   │   ├── ./models/backbones/mobilefacenet.py
│   │   └── ./models/backbones/partial_fc.py
│   ├── ./models/heads
│   │   ├── ./models/heads/adaface_head.py
│   │   ├── ./models/heads/arcface_head.py
│   │   └── ./models/heads/cosface_head.py
│   ├── ./models/inference
│   │   ├── ./models/inference/embedding_model.py
│   │   ├── ./models/inference/face_aligner.py
│   │   └── ./models/inference/face_detector.py
│   ├── ./models/__init__.py
│   └── ./models/losses
│       ├── ./models/losses/arcface_loss.py
│       └── ./models/losses/triplet_loss.py
├── ./preprocess
│   ├── ./preprocess/align_face.py
│   ├── ./preprocess/blur_detector.py
│   ├── ./preprocess/brightness_checker.py
│   ├── ./preprocess/build_dataset.py
│   ├── ./preprocess/crop_face.py
│   ├── ./preprocess/detect_face.py
│   ├── ./preprocess/generate_pairs.py
│   ├── ./preprocess/__init__.py
│   └── ./preprocess/pose_validator.py
├── ./serving
│   ├── ./serving/__init__.py
│   └── ./serving/realtime_inference.py
├── ./training
│   ├── ./training/full_train
│   └── ./training/pretrained
│       ├── ./training/pretrained/build_faiss_indexes.py
│       ├── ./training/pretrained/generate_embeddings.py
│       └── ./training/pretrained/_init__.py
├── ./ui
│   ├── ./ui/control_panel.py
│   ├── ./ui/__init__.py
│   ├── ./ui/main_window.py
│   ├── ./ui/progress_panel.py
│   └── ./ui/webcam_panel.py
└── ./utils
    ├── ./utils/config.py
    └── ./utils/__init__.py

18 directories, 44 files
`
