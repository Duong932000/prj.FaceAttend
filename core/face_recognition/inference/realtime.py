import cv2
import yaml
from pathlib import Path
from inference.pipeline import FacePipeline

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/config.yml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

pipeline = FacePipeline(config)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pipeline.run(frame)

    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        label = f"{r['name']} ({r['score']:.2f})"

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

