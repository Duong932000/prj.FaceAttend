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


import cv2
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from models.MobileNetV3 import get_model


# =========================
# CONFIG
# =========================
CLASS_NAMES = [
    "no_mask",
    "mask",
    "mask_chin",
    "mask_mouth_chin",
    "mask_nose_mouth",
]

INPUT_SIZE = 112

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(__file__).resolve().parents[3] / "output" / "face_mask_detection" / "face_mask_model.pth"


# =========================
# LOAD MODEL
# =========================
model = get_model(num_classes=len(CLASS_NAMES), pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# =========================
# PREPROCESS (MATCH TRAINING)
# =========================
def preprocess(img):
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = img / 255.0

    # normalize giống ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    return torch.tensor(img, dtype=torch.float32).to(DEVICE)


# =========================
# REALTIME INFERENCE
# =========================
def run():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # ---- NOTE:
        # hiện tại bạn chưa detect face → dùng full frame
        input_tensor = preprocess(frame)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)

            confidence, pred = torch.max(probs, dim=1)

            pred = pred.item()
            confidence = confidence.item()

        label = CLASS_NAMES[pred]

        # =========================
        # DRAW UI
        # =========================
        text = f"{label} ({confidence:.2f})"

        # màu theo class
        color_map = {
            "no_mask": (0, 0, 255),
            "mask": (0, 255, 0),
            "mask_chin": (0, 255, 255),
            "mask_mouth_chin": (255, 0, 255),
            "mask_nose_mouth": (255, 255, 0),
        }

        color = color_map.get(label, (255, 255, 255))

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Face Mask Detection - Realtime", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run()


