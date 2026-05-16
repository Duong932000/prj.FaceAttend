
import time
import threading

from insightface.app import FaceAnalysis
from core.face_recognition.utils.config import load_enrollment_config
from core.face_recognition.enrollment.processing.pose_validator import PoseValidator
from core.face_recognition.enrollment.processing.stability_tracker import StabilityTracker
from core.face_recognition.enrollment.processing.quality_assessor import QualityAssessor

class FaceProcessor:
    def __init__(self, camera_stream):

        _, self.cfg = load_enrollment_config()
        self.camera_stream = camera_stream

        self.latest_result = None
        self.target_pose = "front"
        self.running = False

        self.pose_validator = PoseValidator(self.cfg["enrollment"]["poses"])
        self.stability_tracker = StabilityTracker(self.cfg["enrollment"]["stability"])
        self.quality_assessor = QualityAssessor(self.cfg["enrollment"]["quality"])

        self.app = FaceAnalysis(name=self.cfg["enrollment"]["face_detection"]["model_name"])
        self.app.prepare(ctx_id=self.cfg["enrollment"]["face_detection"]["ctx_id"],
                         det_size=tuple(self.cfg["enrollment"]["face_detection"]["det_size"]))

    def start(self):

        self.running = True

        threading.Thread(target=self.process_loop, daemon=True,).start()

    def process_loop(self):

        while self.running:
            frame = self.camera_stream.get_latest_frame()
            if frame is None:
                continue

            faces = self.app.get(frame)

            result = {
                "frame": frame,
                "face_detected": False,
                "stable": False,
                "pose_valid": False,
                "quality": None,
                "face": None,
            }

            if len(faces) > 0:
                face = faces[0]
                result["face_detected"] = True
                result["face"] = face
                x1, y1, x2, y2 = face.bbox.astype(int)
                face_crop = frame[y1:y2, x1:x2]

                quality = self.quality_assessor.evaluate(face_crop)
                self.stability_tracker.update(face)

                stable = self.stability_tracker.is_stable()

                pose_valid = (self.pose_validator.validate(face, self.target_pose))

                result["quality"] = quality
                result["stable"] = stable
                result["pose_valid"] = pose_valid

            self.latest_result = result

            time.sleep(self.cfg["enrollment"]["performance"]["inference_sleep_sec"])