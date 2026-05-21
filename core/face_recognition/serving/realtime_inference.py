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
import json
import time
import faiss
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from core.face_recognition.utils.load_configs import load_common_config

class RealtimeInference:
    def __init__(self):

        # config
        root_dir, config = load_common_config()
        self.cfg = config["face_recognition"]
        self.inference_cfg = self.cfg["realtime_inference"]

        self.camera_id = (self.inference_cfg["camera_id"])
        self.window_name = (self.inference_cfg["window_name"])
        self.det_size = tuple(self.inference_cfg["det_size"])
        self.similarity_threshold = (self.inference_cfg["similarity_threshold"])
        self.show_fps = (self.inference_cfg["show_fps"])
        self.draw_landmarks = (self.inference_cfg["draw_landmarks"])
        self.unknown_label = (self.inference_cfg["unknown_label"])
        self.frame_width = (self.inference_cfg["frame_width"])
        self.frame_height = (self.inference_cfg["frame_height"])

        # path
        self.embedding_dir = Path(root_dir / self.cfg["paths"]["embeddings"])
        self.index_dir = Path(root_dir / self.cfg["paths"]["indexes"])
        self.labels_path = self.embedding_dir / "labels.json"
        self.index_path = self.index_dir / "face_recognition.index"

        self.labels = self.load_labels()

        self.index = self.load_faiss_index()

        self.app = self.load_insightface()

        self.cap = self.initialize_camera()

        self.prev_time = time.time()

    def load_labels(self):

        if not self.labels_path.exists():

            raise FileNotFoundError(f"labels.json not found: {self.labels_path}")

        with open(self.labels_path, "r") as f:
            labels = json.load(f)

        print(f"[INFO] Loaded labels: {len(labels)}")

        return labels

    def load_faiss_index(self):

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        index = faiss.read_index(str(self.index_path))

        print(f"[INFO] Loaded FAISS index: {index.ntotal}")

        return index

    def load_insightface(self):

        print("[INFO] Loading InsightFace...")

        app = FaceAnalysis(name=self.cfg["model"]["name"])

        app.prepare(ctx_id=self.cfg["model"]["ctx_id"], det_size=self.det_size)

        print("[INFO] InsightFace loaded")

        return app

    def initialize_camera(self):

        cap = cv2.VideoCapture(self.camera_id)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)

        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        return cap

    def get_identity(self, embedding):

        embedding = embedding.astype(np.float32)

        embedding = embedding.reshape(1, -1)

        faiss.normalize_L2(embedding)

        similarities, indices = (self.index.search(embedding, 1))

        similarity = float(similarities[0][0])

        matched_idx = int(indices[0][0])

        if (matched_idx >= 0 and similarity >= self.similarity_threshold):
            identity = self.labels[matched_idx]
            color = (0, 255, 0)
        else:
            identity = (self.unknown_label)
            color = (0, 0, 255)

        return (identity, similarity, color)

    def draw_face_info(self, frame, bbox, identity, similarity, color):

        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = (f"{identity}, ({similarity:.2f})")

        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    def draw_face_landmarks(self, frame, face):

        if not self.draw_landmarks:
            return

        for point in face.kps:
            px, py = map(int, point)
            cv2.circle(frame, (px, py), 2, (255, 255, 0), -1)

    def draw_fps(self, frame):

        if not self.show_fps:
            return

        current_time = time.time()

        fps = 1.0 / (current_time - self.prev_time)

        self.prev_time = current_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    def process_frame(self, frame):

        faces = self.app.get(frame)
        for face in faces:
            try:
                (identity, similarity, color) = self.get_identity(face.embedding)
                self.draw_face_info(frame, face.bbox, identity, similarity, color)
                self.draw_face_landmarks(frame, face)
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

        self.draw_fps(frame)

        return frame

    def run(self):

        print("[INFO] Starting realtime inference...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = self.process_frame(frame)

            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        self.cleanup()

    def cleanup(self):

        self.cap.release()

        cv2.destroyAllWindows()

        print("[INFO] Realtime inference stopped")

if __name__ == "__main__":

    RealtimeInference().run()
