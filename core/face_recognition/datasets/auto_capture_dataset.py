# core/face_recognition/datasets/auto_capture_dataset.py

import os
import cv2
import time
import numpy as np

from pathlib import Path
from insightface.app import FaceAnalysis


# =========================================================
# CONFIG
# =========================================================

PERSON_NAME = "duong"

CAMERA_ID = 0

IMAGE_SIZE = (112, 112)

CAPTURE_DELAY_SEC = 0.5

IMAGES_PER_POSE = 15

MIN_FACE_SIZE = 140
MIN_BLUR_SCORE = 80
MIN_BRIGHTNESS = 45

STABLE_FRAMES_REQUIRED = 12

WINDOW_NAME = "Face Recognition Dataset Capture"

POSES = [
    ("neutral", "Look Straight"),
    ("left", "Turn Left"),
    ("right", "Turn Right"),
    ("up", "Look Up"),
    ("down", "Look Down"),
    ("glasses", "Wear Glasses"),
    ("mask", "Wear Mask"),
]


# =========================================================
# ROOT DIRECTORY
# =========================================================

ROOT_DIR = Path(
    os.getenv("FACE_ATTEND_ROOT", ".")
).resolve()

OUTPUT_DIR = (
    ROOT_DIR
    / "core"
    / "face_recognition"
    / "datasets"
    / "raw"
    / PERSON_NAME
)

print(f"[INFO] ROOT_DIR: {ROOT_DIR}")
print(f"[INFO] OUTPUT_DIR: {OUTPUT_DIR}")


# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pose_name, _ in POSES:
    (OUTPUT_DIR / pose_name).mkdir(
        parents=True,
        exist_ok=True,
    )


# =========================================================
# LOAD INSIGHTFACE
# =========================================================

print("[INFO] Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_s",
    allowed_modules=["detection"]
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320),
)

print("[INFO] InsightFace loaded")


# =========================================================
# UTILS
# =========================================================

def draw_text(
    frame,
    text,
    y,
    color=(0, 255, 0),
    scale=0.8,
):

    cv2.putText(
        frame,
        text,
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def compute_blur_score(face_crop):

    gray = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2GRAY,
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()


def compute_brightness(face_crop):

    hsv = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2HSV,
    )

    return hsv[:, :, 2].mean()


def estimate_pose(face):

    landmarks = face.kps

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    eye_center_x = (
        left_eye[0] + right_eye[0]
    ) / 2

    nose_offset_x = nose[0] - eye_center_x

    if nose_offset_x < -15:
        return "left"

    if nose_offset_x > 15:
        return "right"

    eye_center_y = (
        left_eye[1] + right_eye[1]
    ) / 2

    nose_offset_y = nose[1] - eye_center_y

    if nose_offset_y < -10:
        return "up"

    if nose_offset_y > 20:
        return "down"

    return "neutral"


def is_face_inside_roi(face_bbox, roi):

    x1, y1, x2, y2 = map(int, face_bbox)

    fx = (x1 + x2) // 2
    fy = (y1 + y2) // 2

    rx1, ry1, rx2, ry2 = roi

    return (
        rx1 < fx < rx2
        and ry1 < fy < ry2
    )


def validate_face(face, frame):

    x1, y1, x2, y2 = map(int, face.bbox)

    w = x2 - x1
    h = y2 - y1

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False, "Move Closer"

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return False, "Invalid Crop"

    blur_score = compute_blur_score(face_crop)

    if blur_score < MIN_BLUR_SCORE:
        return False, "Blurred"

    brightness = compute_brightness(face_crop)

    if brightness < MIN_BRIGHTNESS:
        return False, "Too Dark"

    return True, "Good"


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(CAMERA_ID)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")


# =========================================================
# SESSION STATE
# =========================================================

current_pose_index = 0
saved_count = 0

stable_frames = 0

is_ready = False

last_capture_time = 0

countdown_started = False
countdown_start_time = None

print("[INFO] Starting dataset capture...")


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    display_frame = frame.copy()

    frame_h, frame_w = frame.shape[:2]

    # =====================================================
    # ROI BOX
    # =====================================================

    roi_w = 320
    roi_h = 420

    roi_x1 = (frame_w - roi_w) // 2
    roi_y1 = (frame_h - roi_h) // 2

    roi_x2 = roi_x1 + roi_w
    roi_y2 = roi_y1 + roi_h

    roi = (
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2,
    )

    cv2.rectangle(
        display_frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (255, 255, 0),
        2,
    )

    # =====================================================
    # GET CURRENT POSE
    # =====================================================

    pose_name, pose_instruction = POSES[
        current_pose_index
    ]

    # =====================================================
    # DRAW UI
    # =====================================================

    draw_text(
        display_frame,
        f"Person: {PERSON_NAME}",
        40,
    )

    draw_text(
        display_frame,
        f"Pose [{current_pose_index + 1}/{len(POSES)}]: {pose_instruction}",
        80,
    )

    draw_text(
        display_frame,
        f"Captured: {saved_count}/{IMAGES_PER_POSE}",
        120,
    )

    # =====================================================
    # DETECT FACE
    # =====================================================

    faces = app.get(frame)

    if len(faces) == 0:

        stable_frames = 0
        is_ready = False

        draw_text(
            display_frame,
            "No Face Detected",
            170,
            (0, 0, 255),
        )

        cv2.imshow(
            WINDOW_NAME,
            display_frame,
        )

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

        continue

    if len(faces) > 1:

        stable_frames = 0
        is_ready = False

        draw_text(
            display_frame,
            "Multiple Faces Detected",
            170,
            (0, 0, 255),
        )

        cv2.imshow(
            WINDOW_NAME,
            display_frame,
        )

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

        continue

    face = faces[0]

    x1, y1, x2, y2 = map(
        int,
        face.bbox,
    )

    detected_pose = estimate_pose(face)

    valid, quality_message = validate_face(
        face,
        frame,
    )

    inside_roi = is_face_inside_roi(
        face.bbox,
        roi,
    )

    # =====================================================
    # STABILITY
    # =====================================================

    if valid and inside_roi:
        stable_frames += 1
    else:
        stable_frames = 0

    stable_ok = (
        stable_frames >= STABLE_FRAMES_REQUIRED
    )

    # =====================================================
    # DRAW FACE
    # =====================================================

    color = (
        (0, 255, 0)
        if valid and inside_roi
        else (0, 0, 255)
    )

    cv2.rectangle(
        display_frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    # =====================================================
    # DRAW STATUS
    # =====================================================

    draw_text(
        display_frame,
        f"Detected Pose: {detected_pose}",
        170,
        color,
    )

    draw_text(
        display_frame,
        f"Quality: {quality_message}",
        210,
        color,
    )

    draw_text(
        display_frame,
        f"Stable Frames: {stable_frames}/{STABLE_FRAMES_REQUIRED}",
        250,
        color,
    )

    if not inside_roi:

        draw_text(
            display_frame,
            "Center Your Face Inside Box",
            290,
            (0, 0, 255),
        )

    # =====================================================
    # READY STATE
    # =====================================================

    if stable_ok and not is_ready:

        draw_text(
            display_frame,
            "Press SPACE When Ready",
            330,
            (0, 255, 255),
        )

    # =====================================================
    # KEYBOARD
    # =====================================================

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    if (
        key == ord(" ")
        and stable_ok
    ):

        is_ready = True

        countdown_started = True

        countdown_start_time = time.time()

    # =====================================================
    # COUNTDOWN
    # =====================================================

    if countdown_started:

        elapsed = (
            time.time()
            - countdown_start_time
        )

        countdown_value = 3 - int(elapsed)

        if countdown_value > 0:

            cv2.putText(
                display_frame,
                str(countdown_value),
                (
                    frame_w // 2 - 40,
                    frame_h // 2,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (0, 255, 255),
                6,
            )

        else:

            countdown_started = False

    # =====================================================
    # CAPTURE
    # =====================================================

    should_capture = (
        valid
        and inside_roi
        and stable_ok
        and is_ready
        and not countdown_started
    )

    current_time = time.time()

    if (
        should_capture
        and current_time - last_capture_time
        > CAPTURE_DELAY_SEC
    ):

        face_crop = frame[
            y1:y2,
            x1:x2,
        ]

        aligned_face = cv2.resize(
            face_crop,
            IMAGE_SIZE,
        )

        filename = (
            OUTPUT_DIR
            / pose_name
            / f"{PERSON_NAME}_{saved_count:03d}.jpg"
        )

        cv2.imwrite(
            str(filename),
            aligned_face,
        )

        print(f"[SAVE] {filename}")

        saved_count += 1

        last_capture_time = current_time

    # =====================================================
    # NEXT POSE
    # =====================================================

    if saved_count >= IMAGES_PER_POSE:

        current_pose_index += 1

        saved_count = 0

        stable_frames = 0

        is_ready = False

        countdown_started = False

        time.sleep(1)

        if current_pose_index >= len(POSES):

            print("[INFO] Dataset capture completed")
            break

    # =====================================================
    # SHOW WINDOW
    # =====================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame,
    )

# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("[INFO] Capture session ended")