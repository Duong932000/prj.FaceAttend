#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      realtime_inference.py
# .:ccccccccccccc;KMMc;cc;xMMc;ccccccc:.    Description  :      
# ,cccccccccccccc;MMM.;cc;;WW:;cccccccc,    --------------
# :cccccccccccccc;MMM.;cccccccccccccccc:
# :ccccccc;oxOOOo;MMM000k.;cccccccccccc:
# cccccc;0MMKxdd:;MMMkddc.;cccccccccccc;
# ccccc;XMO';cccc;MMM.;cccccccccccccccc'
# ccccc;MMo;ccccc;MMW.;ccccccccccccccc;
# ccccc;0MNc.ccc.xMMd;ccccccccccccccc;
# cccccc;dNMWXXXWM0:;cccccccccccccc:,
# cccccccc;.:odl:.;cccccccccccccc:,.
# ccccccccccccccccccccccccccccc:'.
# :ccccccccccccccccccccccc:;,..
#  ':cccccccccccccccc::;,.
#########################################################


import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path

from core.face_mask_detection.models.MobileNetV3 import get_model

CLASS_NAMES = [
    'mask',
    'mask_chin',
    'mask_mouth_chin',
    'mask_nose_mouth',
    'no_mask'
]

INPUT_SIZE = 112
CONF_THRESHOLD = 0.50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError(f"Failed to load HaarCascade: {CASCADE_PATH}")

def load_model(model_output_path):

    model = get_model(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False,
        width_mult=1.0
    )

    state_dict = torch.load(model_output_path, map_location=device)

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model

def preprocess(face_img):

    if face_img is None:
        return None

    face_img = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    face_img = face_img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    face_img = (face_img - mean) / std

    face_img = np.transpose(face_img, (2, 0, 1))

    face_img = np.expand_dims(face_img, axis=0)

    tensor = torch.tensor(face_img, dtype=torch.float32).to(device)

    return tensor

def predict(model, face_img):

    input_tensor = preprocess(face_img)

    if input_tensor is None:
        return None

    with torch.no_grad():

        output = model(input_tensor)

        probs = F.softmax(output, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    return {
        "label": CLASS_NAMES[pred.item()],
        "class_id": pred.item(),
        "confidence": confidence.item()
    }

def draw_prediction(frame, box, result):

    x1, y1, x2, y2 = box

    label = result["label"]
    conf = result["confidence"]

    if label == "mask":
        color = (0, 255, 0)

    elif label == "no_mask":
        color = (0, 0, 255)

    else:
        color = (0, 165, 255)

    text = f"{label}: {conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), color, -1)

    cv2.putText(
        frame,
        text,
        (x1 + 5, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

def detect_faces(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    return faces

def main():

    # Linux/OpenCV GUI fix
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ["DISPLAY"] = ":0"

    ROOT_DIR = Path(__file__).resolve().parents[3]

    model_output_path = (
        ROOT_DIR
        / "output"
        / "face_mask_detection"
        / "pth"
        / "face_mask_model.pth"
    )

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading model: {model_output_path}")

    if not model_output_path.exists():
        print(f"[ERROR] Model not found")
        return

    model = load_model(model_output_path)

    print("[INFO] Model loaded successfully")

    # webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Press 'q' to quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("[WARN] Failed to read frame")
            break

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:

            # margin
            margin = 20

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)

            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            result = predict(model, face_crop)

            if result is None:
                continue

            if result["confidence"] < CONF_THRESHOLD:
                continue

            draw_prediction(
                frame,
                (x1, y1, x2, y2),
                result
            )

        cv2.imshow("Realtime Face Mask Detection", frame)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

    print("[INFO] Webcam closed")

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    except Exception as e:
        raise e