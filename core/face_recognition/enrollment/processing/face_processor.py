#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      face_processor.py
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


import time
import threading

from insightface.app import FaceAnalysis
from core.face_recognition.utils.config import load_enrollment_config
from core.face_recognition.enrollment.processing.pose_validator import PoseValidator
from core.face_recognition.enrollment.processing.quality_assessor import QualityAssessor
from core.face_recognition.enrollment.processing.stability_tracker import StabilityTracker

class FaceProcessor:
    def __init__(self, camera_stream):

        _, self.cfg = load_enrollment_config()
        self.camera_stream = camera_stream

        self.latest_result = None
        self.target_pose = "front"
        self.running = False
        self.detection_enabled = False

        self.pose_validator = PoseValidator(self.cfg["enrollment"]["poses"])
        self.stability_tracker = StabilityTracker(self.cfg["enrollment"]["stability"])
        self.quality_assessor = QualityAssessor(self.cfg["enrollment"]["quality"])

        self.app = FaceAnalysis(name=self.cfg["enrollment"]["face_detection"]["model_name"])
        self.app.prepare(ctx_id=self.cfg["enrollment"]["face_detection"]["ctx_id"],
                         det_size=tuple(self.cfg["enrollment"]["face_detection"]["det_size"]))

    def start(self):

        self.running = True

        threading.Thread(target=self.process_loop, daemon=True,).start()

    def enable_detection(self):

        self.detection_enabled = True

    def disable_detection(self):

        self.detection_enabled = False

        self.latest_result = None

    def process_loop(self):

        while self.running:
            if not self.detection_enabled:
                time.sleep(0.1)
                continue
            frame = self.camera_stream.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
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

            # No face detect
            if len(faces) == 0:
                self.latest_result = result
                time.sleep(self.cfg["enrollment"]["performance"]["inference_sleep_sec"])
                continue

            face = faces[0]
            result["face_detected"] = True
            result["face"] = face

            # face box
            x1, y1, x2, y2 = face.bbox.astype(int)
            h, w, _ = frame.shape

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w, x2)
            y2 = min(h, y2)

            # face crop
            face_crop = frame[y1:y2, x1:x2]

            # invalid face crop
            if face_crop.size == 0:
                self.latest_result = result
                time.sleep(self.cfg["enrollment"]["performance"]["inference_sleep_sec"])
                continue

            quality = self.quality_assessor.evaluate(face_crop)
            result["quality"] = quality

            self.stability_tracker.update(face)

            stable = self.stability_tracker.is_stable()
            result["stable"] = stable
            
            # pose validation
            pose_valid = self.pose_validator.validate(face,self.target_pose)
            result["pose_valid"] = pose_valid

            # save result
            self.latest_result = result

            # control FPS
            time.sleep(self.cfg["enrollment"]["performance"]["inference_sleep_sec"])