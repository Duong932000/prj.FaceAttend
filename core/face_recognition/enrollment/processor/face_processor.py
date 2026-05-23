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
from core.face_recognition.utils.load_configs import load_enrollment_config
from core.face_recognition.enrollment.processor.pose_validator import PoseValidator
from core.face_recognition.enrollment.processor.quality_assessor import QualityAssessor
from core.face_recognition.enrollment.processor.stability_tracker import StabilityTracker


class FaceProcessor:
    def __init__(self, camera_stream):

        self.enroll_camera_cfg, \
        self.enroll_face_detection_cfg, \
        self.enroll_quality_cfg, \
        self.enroll_stability_cfg, _, _, \
        self.enroll_performance_cfg, _, _, _ \
            = load_enrollment_config()

        self.camera_stream = camera_stream

        self.target_pose = "front"
        self.latest_result = None
        self.running = False
        self.detection_enabled = False

        self.result_lock = threading.Lock()

        # modules
        self.pose_validator = PoseValidator(self.enroll_camera_cfg)
        self.stability_tracker = StabilityTracker(self.enroll_stability_cfg)
        self.quality_assessor = QualityAssessor(self.enroll_quality_cfg)

        # Face analysis
        self.face_analysis = FaceAnalysis(name=self.enroll_face_detection_cfg["model_name"])

        self.face_analysis.prepare(ctx_id=self.enroll_face_detection_cfg["ctx_id"],
                                   det_size=tuple(self.enroll_face_detection_cfg["det_size"]))

        self.inference_fps = 0.0
        self._fps_alpha = 0.1
        self._last_infer_time = None

    def start(self):

        self.running = True
        threading.Thread(target=self.process_loop, daemon=True).start()

    def stop(self):

        self.running = False

    def enable_detection(self):

        self.detection_enabled = True

    def disable_detection(self):

        self.detection_enabled = False

        with self.result_lock:
            self.latest_result = None

        self.stability_tracker.reset()
        self.inference_fps = 0.0
        self._last_infer_time = None

    def get_latest_result(self):

        with self.result_lock:

            if self.latest_result is None:
                return None

            return self.latest_result.copy()

    def update_inference_fps(self):

        now = time.time()
        if self._last_infer_time is not None:
            dt = now - self._last_infer_time
            if dt > 0:
                instant_fps = 1.0 / dt
                if self.inference_fps <= 0:
                    self.inference_fps = instant_fps
                else:
                    self.inference_fps = ((1 - self._fps_alpha) * self.inference_fps + self._fps_alpha * instant_fps)
        
        self._last_infer_time = now

    def process_loop(self):

        inference_sleep = self.enroll_performance_cfg["inference_sleep_sec"]

        while self.running:
            if not self.detection_enabled:
                time.sleep(0.1)
                continue

            frame = self.camera_stream.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            faces = self.face_analysis.get(frame)
            self.update_inference_fps()

            result = {
                "frame": frame,
                "face_detected": False,
                "stable": False,
                "pose_valid": False,
                "quality": None,
                "face": None,
            }

            if len(faces) == 0:
                self.stability_tracker.reset()
                with self.result_lock:
                    self.latest_result = result
                time.sleep(inference_sleep)
                continue

            face = faces[0]
            result["face_detected"] = True
            result["face"] = face

            x1, y1, x2, y2 = face.bbox.astype(int)

            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # min, max size check
            face_width = x2 - x1
            face_height = y2 - y1

            if face_width < 120 or face_height < 120:
                with self.result_lock:
                    self.latest_result = result
                time.sleep(inference_sleep)
                continue

            # face crop
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                with self.result_lock:
                    self.latest_result = result
                time.sleep(inference_sleep)
                continue

            result["quality"] = self.quality_assessor.evaluate(face_crop)
            self.stability_tracker.update(face)
            result["stable"] = self.stability_tracker.is_stable()
            result["pose_valid"] = self.pose_validator.validate(face, self.target_pose)

            # save result
            with self.result_lock:
                self.latest_result = result

            time.sleep(inference_sleep)
