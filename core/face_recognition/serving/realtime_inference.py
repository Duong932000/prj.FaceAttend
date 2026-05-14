# core/face_recognition/serving/realtime_inference.py

import os
import cv2
import json
import time
import faiss
import numpy as np

from pathlib import Path

from insightface.app import FaceAnalysis


# =========================================================
# CONFIG
# =========================================================

CAMERA_ID = 0

WINDOW_NAME = "Realtime Face Recognition"

DET_SIZE = (320, 320)

SIMILARITY_THRESHOLD = 0.45

SHOW_FPS = True

UNKNOWN_LABEL = "UNKNOWN"

DRAW_LANDMARKS = False


# =========================================================
# ROOT DIRECTORY
# =========================================================

ROOT_DIR = Path(
    os.getenv("FACE_ATTEND_ROOT", ".")
).resolve()

EMBEDDING_DIR = (
    ROOT_DIR
    / "core"
    / "face_recognition"
    / "datasets"
    / "embeddings"
)

INDEX_DIR = (
    ROOT_DIR
    / "core"
    / "face_recognition"
    / "datasets"
    / "indexes"
)

print(f"[INFO] ROOT_DIR: {ROOT_DIR}")


# =========================================================
# LOAD LABELS
# =========================================================

labels_path = (
    EMBEDDING_DIR
    / "labels.json"
)

if not labels_path.exists():

    raise FileNotFoundError(
        f"labels.json not found: {labels_path}"
    )

with open(labels_path, "r") as f:

    labels = json.load(f)

print(f"[INFO] Loaded labels: {len(labels)}")


# =========================================================
# LOAD FAISS INDEX
# =========================================================

index_path = (
    INDEX_DIR
    / "face_recognition.index"
)

if not index_path.exists():

    raise FileNotFoundError(
        f"FAISS index not found: {index_path}"
    )

index = faiss.read_index(
    str(index_path)
)

print(
    f"[INFO] Loaded FAISS index: {index.ntotal}"
)


# =========================================================
# LOAD INSIGHTFACE
# =========================================================

print("[INFO] Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_s",
)

app.prepare(
    ctx_id=0,
    det_size=DET_SIZE,
)

print("[INFO] InsightFace loaded")


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(CAMERA_ID)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():

    raise RuntimeError(
        "Cannot open webcam"
    )


# =========================================================
# FPS
# =========================================================

prev_time = time.time()


# =========================================================
# DRAW FUNCTION
# =========================================================

def draw_face_info(
    frame,
    bbox,
    identity,
    similarity,
    color,
):

    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    label = (
        f"{identity} "
        f"({similarity:.2f})"
    )

    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


# =========================================================
# MAIN LOOP
# =========================================================

print("[INFO] Starting realtime inference...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    display_frame = frame.copy()

    # =====================================================
    # FACE DETECTION
    # =====================================================

    faces = app.get(frame)

    # =====================================================
    # PROCESS FACES
    # =====================================================

    for face in faces:

        try:

            # =============================================
            # GET EMBEDDING
            # =============================================

            embedding = face.embedding.astype(
                np.float32
            )

            embedding = embedding.reshape(1, -1)

            faiss.normalize_L2(
                embedding
            )

            # =============================================
            # SEARCH FAISS
            # =============================================

            similarities, indices = index.search(
                embedding,
                1,
            )

            similarity = float(
                similarities[0][0]
            )

            matched_idx = int(
                indices[0][0]
            )

            # =============================================
            # MATCH LABEL
            # =============================================

            if (
                matched_idx >= 0
                and similarity
                >= SIMILARITY_THRESHOLD
            ):

                identity = labels[
                    matched_idx
                ]

                color = (0, 255, 0)

            else:

                identity = UNKNOWN_LABEL

                color = (0, 0, 255)

            # =============================================
            # DRAW RESULT
            # =============================================

            draw_face_info(
                display_frame,
                face.bbox,
                identity,
                similarity,
                color,
            )

            # =============================================
            # OPTIONAL LANDMARKS
            # =============================================

            if DRAW_LANDMARKS:

                for point in face.kps:

                    px, py = map(
                        int,
                        point,
                    )

                    cv2.circle(
                        display_frame,
                        (px, py),
                        2,
                        (255, 255, 0),
                        -1,
                    )

        except Exception as e:

            print(f"[ERROR] {e}")

            continue

    # =====================================================
    # FPS
    # =====================================================

    if SHOW_FPS:

        current_time = time.time()

        fps = 1.0 / (
            current_time - prev_time
        )

        prev_time = current_time

        cv2.putText(
            display_frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # =====================================================
    # SHOW WINDOW
    # =====================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame,
    )

    # =====================================================
    # EXIT
    # =====================================================

    key = cv2.waitKey(1)

    if key == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("[INFO] Realtime inference stopped")