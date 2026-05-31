import cv2
import torch

from pathlib import Path

from core.face_mask_detection.models.MobileNetV3 import get_model
from core.face_mask_detection.detectors.scrfd_detector import SCRFDDetector
from core.face_mask_detection.inference.preprocess import FacePreprocessor
from core.face_mask_detection.inference.classifier import MaskClassifier
from core.face_mask_detection.business.risk_mapper import map_risk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):

    model = get_model(
        num_classes=5,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False,
        width_mult=1.0
    )

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.to(DEVICE)

    model.eval()

    return model

def draw_result(frame, bbox, prediction, risk_status):

    x1, y1, x2, y2 = bbox

    label = prediction["class_name"]
    conf = prediction["confidence"]

    if risk_status == "safe":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    text = f"{label} ({risk_status}) {conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        frame,
        text,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

def main():

    root_dir = Path(__file__).resolve().parents[3]

    model_path = (
        root_dir
        / "output"
        / "face_mask_detection"
        / "pth"
        / "face_mask_model.pth"
    )

    print(f"[INFO] Loading model: {model_path}")

    model = load_model(model_path)

    detector = SCRFDDetector()

    preprocessor = FacePreprocessor(image_size=112)

    classifier = MaskClassifier(
        model=model,
        preprocessor=preprocessor,
        device=DEVICE
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            prediction = classifier.predict(face_crop)

            risk_status = map_risk(prediction["class_name"])

            draw_result(
                frame,
                (x1, y1, x2, y2),
                prediction,
                risk_status
            )

        cv2.imshow("Realtime Face Mask Detection", frame)

        key = cv2.waitKey(1)

        if key & 0xFF == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()