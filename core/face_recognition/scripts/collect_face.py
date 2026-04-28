import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from insightface.app import FaceAnalysis

USER_NAME = "duong"
SAVE_DIR = Path(__file__).resolve().parents[2] / "database/raw" / USER_NAME

MAX_SAMPLES = 30
MIN_FACE_SIZE = 100
BLUR_THRESHOLD = 100
SIMILARITY_THRESHOLD = 0.6

DEVICE = 0  # webcam

SAVE_DIR.mkdir(parents=True, exist_ok=True)

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # GPU, dùng -1 nếu CPU

cap = cv2.VideoCapture(DEVICE)

saved_embeddings = []
count = 0

def is_blurry(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < BLUR_THRESHOLD

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_new_pose(emb):
    for e in saved_embeddings:
        if cosine_similarity(e, emb) > SIMILARITY_THRESHOLD:
            return False
    return True

print("Start collecting faces... Press ESC to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        w, h = x2 - x1, y2 - y1

        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        if is_blurry(face_crop):
            continue

        emb = face.embedding

        if not is_new_pose(emb):
            continue

        # SAVE IMAGE
        filename = f"{datetime.now().strftime('%H%M%S_%f')}.jpg"
        save_path = SAVE_DIR / filename

        cv2.imwrite(str(save_path), face_crop)

        saved_embeddings.append(emb)
        count += 1

        print(f"Saved {count}/{MAX_SAMPLES}: {filename}")

        if count >= MAX_SAMPLES:
            break

    # DRAW UI
    cv2.putText(frame, f"Collected: {count}/{MAX_SAMPLES}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Collect Faces", frame)

    if cv2.waitKey(1) & 0xFF == 27 or count >= MAX_SAMPLES:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Done. Saved at: {SAVE_DIR}")