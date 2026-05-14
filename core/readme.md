core/
└── face_recognition/
    ├── datasets/
    │   ├── raw/
    │   ├── aligned/
    │   ├── embeddings/
    │   ├── indexes/
    │   └── splits/
    │
    ├── preprocess/
    │   ├── align_faces.py
    │   ├── create_splits.py
    │   └── augmentations.py
    │
    ├── models/
    │   ├── backbones/
    │   │   ├── iresnet.py
    │   │   └── mobilefacenet.py
    │   │
    │   ├── heads/
    │   │   └── arcface_head.py
    │   │
    │   ├── losses/
    │   │   └── arcface_loss.py
    │   │
    │   └── checkpoints/
    │       ├── arcface_pretrained.onnx
    │       ├── best.pt
    │       └── best.onnx
    │
    ├── training/
    │   ├── datasets/
    │   │   └── face_dataset.py
    │   │
    │   ├── pretrained/
    │   │   ├── generate_embeddings.py
    │   │   ├── build_faiss_index.py
    │   │   └── search_embedding.py
    │   │
    │   ├── full_train/
    │   │   ├── train_arcface.py
    │   │   ├── trainer.py
    │   │   ├── validate.py
    │   │   └── export_embeddings.py
    │   │
    │   └── utils/
    │       ├── metrics.py
    │       └── transforms.py
    │
    ├── serving/
    │   └── realtime_inference.py
    │
    └── export/
        ├── export_onnx.py
        └── export_tensorrt.py