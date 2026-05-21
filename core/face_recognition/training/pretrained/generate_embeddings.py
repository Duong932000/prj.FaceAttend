#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      generate_embeddings.py
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
import json
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from core.face_recognition.utils.load_configs import load_common_config


class EmbeddingsGeneration:
    def __init__(self):

        # load config
        root_dir, config = load_common_config()
        self.cfg = config["face_recognition"]

        # path
        self.aligned_dir = Path(root_dir / self.cfg["paths"]["aligned_faces"])
        self.embedding_dir = Path(root_dir / self.cfg["paths"]["embeddings"])

        self.embedding_dir.mkdir(parents=True, exist_ok=True)

        # model config
        self.model_name = self.cfg["model"]["name"]
        self.det_size = tuple(self.cfg["model"]["det_size"])
        self.ctx_id = self.cfg["model"]["ctx_id"]

        # storages
        self.embeddings = []
        self.labels = []
        self.image_paths = []

        self.total_images = 0
        self.success_images = 0
        self.failed_images = 0

        self.app = self.load_model()

    def load_model(self):

        print("[INFO] Loading InsightFace...")

        app = FaceAnalysis(name=self.model_name)

        app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)

        print("[INFO] InsightFace loaded")

        return app

    def get_largest_face(self, faces):

        return max(faces, key=lambda f: ((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))

    def normalize_embedding(self, embedding):

        embedding = embedding.astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None

        embedding = embedding / norm

        return embedding

    def process_image(self, image_path, label):

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARNING] Cannot read: {image_path}")
            return False

        faces = self.app.get(image)
        if len(faces) == 0:
            print(f"[WARNING] No face: {image_path.name}")
            return False

        face = self.get_largest_face(faces)

        embedding = face.embedding
        if embedding is None:
            print(f"[WARNING] Empty embedding: {image_path.name}")
            return False

        embedding = self.normalize_embedding(embedding)
        if embedding is None:
            print(f"[WARNING] Zero norm embedding: {image_path.name}")
            return False

        self.embeddings.append(embedding)

        self.labels.append(label)

        self.image_paths.append(str(image_path))

        print(f"[EMBED] {image_path.name}")

        return True

    def save_outputs(self):

        if len(self.embeddings) == 0:

            raise RuntimeError("No embeddings generated")

        embeddings = np.array(self.embeddings, dtype=np.float32)

        print(f"\n[INFO] Embedding Shape: {embeddings.shape}")

        embedding_path = (self.embedding_dir / "face_embeddings.npy")

        labels_path = (self.embedding_dir / "labels.json")

        image_paths_path = (self.embedding_dir / "image_paths.json")

        np.save(embedding_path, embeddings)

        with open(labels_path, "w") as f:
            json.dump(self.labels, f, indent=4)

        with open(image_paths_path, "w") as f:
            json.dump(self.image_paths, f, indent=4)

        print(f"\n[SAVED] {embedding_path}")
        print(f"[SAVED] {labels_path}")
        print(f"[SAVED] {image_paths_path}")

    def process_dataset(self):

        print(f"[INFO] ALIGNED_DIR: {self.aligned_dir}")

        print(f"[INFO] EMBEDDING_DIR: {self.embedding_dir}")

        for person_dir in self.aligned_dir.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name

            print(f"\n[INFO] Processing: {person_name}")

            for image_path in person_dir.rglob("*.jpg"):
                self.total_images += 1
                try:
                    success = self.process_image(image_path=image_path, label=person_name)
                    if success:
                        self.success_images += 1
                    else:
                        self.failed_images += 1
                except Exception as e:
                    self.failed_images += 1
                    print(f"[ERROR] {image_path.name}: {e}")
                    continue

        self.save_outputs()

        print("\n========================================")
        print("[INFO] EMBEDDING GENERATION FINISHED")
        print("========================================")

        print(f"Total Images   : {self.total_images}")
        print(f"Success Images : {self.success_images}")
        print(f"Failed Images  : {self.failed_images}")

if __name__ == "__main__":

    EmbeddingsGeneration().process_dataset()