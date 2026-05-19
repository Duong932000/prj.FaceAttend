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

from core.face_recognition.enrollment.processor.pose_validator import PoseValidator
from core.face_recognition.enrollment.processor.quality_assessor import QualityAssessor
from core.face_recognition.enrollment.processor.stability_tracker import StabilityTracker


class FaceProcessor:

    def __init__(self, camera_stream):

        _, self.cfg = load_enrollment_config()

        self.camera_stream = camera_stream

        self.latest_result = None

        self.result_lock = threading.Lock()

        self.target_pose = "front"

        self.running = False

        self.detection_enabled = False

        # =====================================================
        # processor MODULES
        # =====================================================
        self.pose_validator \
            = PoseValidator(
                self.cfg["enrollment"]["poses"]
            )

        self.stability_tracker \
            = StabilityTracker(
                self.cfg["enrollment"]["stability"]
            )

        self.quality_assessor \
            = QualityAssessor(
                self.cfg["enrollment"]["quality"]
            )

        # =====================================================
        # INSIGHTFACE
        # =====================================================
        self.app \
            = FaceAnalysis(
                name=self.cfg["enrollment"]["face_detection"]["model_name"]
            )

        self.app.prepare(
            ctx_id=self.cfg["enrollment"]["face_detection"]["ctx_id"],
            det_size=tuple(
                self.cfg["enrollment"]["face_detection"]["det_size"]
            )
        )

    # =========================================================
    # START PROCESSOR
    # =========================================================
    def start(self):

        self.running = True

        threading.Thread(
            target=self.process_loop,
            daemon=True
        ).start()

    # =========================================================
    # STOP PROCESSOR
    # =========================================================
    def stop(self):

        self.running = False

    # =========================================================
    # ENABLE DETECTION
    # =========================================================
    def enable_detection(self):

        self.detection_enabled = True

    # =========================================================
    # DISABLE DETECTION
    # =========================================================
    def disable_detection(self):

        self.detection_enabled = False

        with self.result_lock:
            self.latest_result = None

        self.stability_tracker.reset()

    # =========================================================
    # GET LATEST RESULT
    # =========================================================
    def get_latest_result(self):

        with self.result_lock:

            if self.latest_result is None:
                return None

            return self.latest_result.copy()

    # =========================================================
    # MAIN PROCESS LOOP
    # =========================================================
    def process_loop(self):

        inference_sleep \
            = self.cfg["enrollment"]["performance"]["inference_sleep_sec"]

        while self.running:

            # =================================================
            # DETECTION DISABLED
            # =================================================
            if not self.detection_enabled:

                time.sleep(0.1)

                continue

            # =================================================
            # GET FRAME
            # =================================================
            frame = self.camera_stream.get_latest_frame()

            if frame is None:

                time.sleep(0.01)

                continue

            # =================================================
            # FACE DETECTION
            # =================================================
            faces = self.app.get(frame)

            result = {
                "frame": frame,
                "face_detected": False,
                "stable": False,
                "pose_valid": False,
                "quality": None,
                "face": None,
            }

            # =================================================
            # NO FACE
            # =================================================
            if len(faces) == 0:

                self.stability_tracker.reset()

                with self.result_lock:
                    self.latest_result = result

                time.sleep(inference_sleep)

                continue

            # =================================================
            # TAKE FIRST FACE
            # =================================================
            face = faces[0]

            result["face_detected"] = True

            result["face"] = face

            # =================================================
            # FACE BOX
            # =================================================
            x1, y1, x2, y2 = face.bbox.astype(int)

            h, w, _ = frame.shape

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w, x2)
            y2 = min(h, y2)

            # =================================================
            # MIN FACE SIZE CHECK
            # =================================================
            face_width = x2 - x1
            face_height = y2 - y1

            if face_width < 120 or face_height < 120:

                with self.result_lock:
                    self.latest_result = result

                time.sleep(inference_sleep)

                continue

            # =================================================
            # FACE CROP
            # =================================================
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:

                with self.result_lock:
                    self.latest_result = result

                time.sleep(inference_sleep)

                continue

            # =================================================
            # QUALITY
            # =================================================
            quality \
                = self.quality_assessor.evaluate(
                    face_crop
                )

            result["quality"] = quality

            # =================================================
            # STABILITY
            # =================================================
            self.stability_tracker.update(face)

            stable \
                = self.stability_tracker.is_stable()

            result["stable"] = stable

            # =================================================
            # POSE VALIDATION
            # =================================================
            pose_valid \
                = self.pose_validator.validate(
                    face,
                    self.target_pose
                )

            result["pose_valid"] = pose_valid

            # =================================================
            # SAVE RESULT
            # =================================================
            with self.result_lock:
                self.latest_result = result

            # =================================================
            # FPS CONTROL
            # =================================================
            time.sleep(inference_sleep)