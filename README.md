
prj.FaceAttend - REAL TIME ATTENDANCE MANAGENMENT SYSTEM BASED ON FACE RECOGNITION WITH FACIAL FEATURES TECHNIQUES

# Overview
Face attendance is a real-time attendance system using computer vision and deep learning. The system identifies individuals through facial recognition, verifies authenticity via liveness detection, and logs attendance data.

# Key Objectives

- Accurate face recognition (including masked face)

- Prevent spoofing (photo/video attacks)

- Real-time performance on edge devices

- Clean, modular, and scalable architecture

- Progressive evolution from desktop - web platform

# System Architecture
`
    Camera Stream (webcam, camera)
        ↓
    Face Detection (InsightFace)
        ↓
    Face Crop Queue (asyncronous)
        ↓
    Processing Pipeline
        ├── Liveness Detection
        ├── Mask Detection
        └── Face Recognition

        ↓
    Decision Engine
        ↓
    Logger
`

# Core Components

- Face Detection & Recognition
    + Detect faces in real-time
    + Extract embeddings
    + Compare against stored embeddings

- Liveness Detection
    + Prevent spoofing using: Photo / Video
    + Output: REAL / FAKE

- Face Mask Detection
    + Mask
    + No Mask

- Decision Engine
    + Combines outputs from all models
    + Determines final attendance status

- Logging System
    + CSV-based logging
    + Structured records for later migration to DB

# Contributor and Maintainer
Nguyen Dac Duong