import os
import json
import numpy as np
import cv2
from pathlib import Path
from models import load_model

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "database/raw"
OUT_EMB = ROOT / "database/embeddings/embeddings.npy"
OUT_META = ROOT / "database/metadata.json"

app = load_model("buffalo_l", "cpu")

embeddings = []
metadata = {}

idx = 0

for person in os.listdir(RAW_DIR):
    person_dir = RAW_DIR / person
    if not person_dir.is_dir():
        continue

    for img_name in os.listdir(person_dir):
        img_path = person_dir / img_name
        img = cv2.imread(str(img_path))

        faces = app.get(img)
        if len(faces) == 0:
            continue

        emb = faces[0].embedding
        embeddings.append(emb)
        metadata[str(idx)] = person
        idx += 1

embeddings = np.array(embeddings)

OUT_EMB.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT_EMB, embeddings)

with open(OUT_META, "w") as f:
    json.dump(metadata, f, indent=2)

print("Database built successfully!")