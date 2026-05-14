#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      faceid_enrollment.py
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
import time

from pathlib import Path

from insightface.app import FaceAnalysis
from core.face_recognition.utils.config import load_config
from core.face_recognition.preprocess.blur_detector import BlurDetector
from core.face_recognition.preprocess.brightness_checker import BrightnessChecker
from core.face_recognition.preprocess.pose_validator import PoseValidator

from core.face_recognition.ui.main_window import MainWindow


class AutoCaptureDataset:
    def __init__(self):

        # load config
        root_dir, config = load_config()
        self.cfg = config["face_recognition"]
        self.capture_cfg = self.cfg["dataset_capture"]

        # paths
        self.raw_dataset_dir = Path(root_dir / self.cfg["paths"]["raw_dataset"])
        self.raw_dataset_dir.mkdir(parents=True, exist_ok=True,)


        self.camera_id = self.capture_cfg["camera_id"]
        self.image_width = self.capture_cfg["image_width"]
        self.image_height = self.capture_cfg["image_height"]
        self.capture_per_pose = self.capture_cfg["capture_per_pose"]
        self.capture_interval = self.capture_cfg["capture_interval_sec"]
        self.blur_threshold = self.capture_cfg["blur_threshold"]
        self.brightness_min = self.capture_cfg["brightness_min"]
        self.brightness_max = self.capture_cfg["brightness_max"]
        self.min_face_size = self.capture_cfg["min_face_size"]
        self.stable_frames_required = self.capture_cfg["stable_frames_required"]
        self.poses = self.capture_cfg["poses"]


        self.person_name = None
        self.capture_started = False
        self.current_pose_index = 0
        self.current_pose_count = 0
        self.last_capture_time = 0
        self.stable_frame_count = 0


        print("[INFO] Loading InsightFace...")
        self.app = FaceAnalysis(name=self.cfg["model"]["name"])
        self.app.prepare(ctx_id=self.cfg["model"]["ctx_id"], det_size=tuple(self.cfg["model"]["det_size"]))

        print("[INFO] InsightFace loaded")

        self.cap = cv2.VideoCapture(self.camera_id)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width,)

        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height,)

        self.window = MainWindow(poses=self.poses, start_callback=self.start_enrollment,)

    def start_enrollment(self, person_name):

        self.person_name = person_name
        self.capture_started = True
        self.window.control_panel.update_status("Enrollment Started")

        print(f"[INFO] Start enrollment: {person_name}")

    def get_current_pose(self):

        if (self.current_pose_index >= len(self.poses)):
            return None

        return self.poses[self.current_pose_index]

    def draw_face_guide(self, frame):

        height, width, _ = frame.shape
        guide_size = 350

        x1 = int(width / 2 - guide_size / 2)

        y1 = int(height / 2 - guide_size / 2)

        x2 = x1 + guide_size
        y2 = y1 + guide_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        return (x1, y1, x2, y2)

    def draw_text(self,frame, text, y, color=(0, 255, 0)):

        cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    def validate_face_centering(self, face_bbox, guide_box):

        fx1, fy1, fx2, fy2 = face_bbox
        gx1, gy1, gx2, gy2 = guide_box
        face_center_x = int(
            (fx1 + fx2) / 2
        )
        face_center_y = int(
            (fy1 + fy2) / 2
        )

        return (
            gx1 < face_center_x < gx2
            and gy1 < face_center_y < gy2
        )

    def save_image(
        self,
        frame,
        pose,
    ):

        output_dir = (
            self.raw_dataset_dir
            / self.person_name
            / pose
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = int(
            time.time() * 1000
        )

        output_path = (
            output_dir
            / f"{timestamp}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame,
        )

        print(
            f"[SAVE] {output_path}"
        )

    def process_frame(
        self,
        frame,
    ):

        guide_box = self.draw_face_guide(
            frame
        )

        if not self.capture_started:

            self.draw_text(
                frame,
                "ENTER NAME AND START ENROLLMENT",
                50,
            )

            return frame

        current_pose = self.get_current_pose()

        if current_pose is None:

            self.draw_text(
                frame,
                "ENROLLMENT COMPLETED",
                50,
            )

            self.window.control_panel.update_status(
                "Enrollment Completed"
            )

            return frame

        self.draw_text(
            frame,
            f"POSE: {current_pose.upper()}",
            50,
        )

        self.draw_text(
            frame,
            (
                f"CAPTURED: "
                f"{self.current_pose_count}/"
                f"{self.capture_per_pose}"
            ),
            100,
        )

        faces = self.app.get(
            frame
        )

        if len(faces) == 0:

            self.stable_frame_count = 0

            self.draw_text(
                frame,
                "NO FACE DETECTED",
                150,
                (0, 0, 255),
            )

            return frame

        face = faces[0]

        x1, y1, x2, y2 = map(
            int,
            face.bbox,
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2,
        )

        face_width = x2 - x1

        if face_width < self.min_face_size:

            self.stable_frame_count = 0

            self.draw_text(
                frame,
                "MOVE CLOSER",
                150,
                (0, 0, 255),
            )

            return frame

        centered = self.validate_face_centering(
            face_bbox=(
                x1,
                y1,
                x2,
                y2,
            ),
            guide_box=guide_box,
        )

        if not centered:

            self.stable_frame_count = 0

            self.draw_text(
                frame,
                "CENTER YOUR FACE",
                150,
                (0, 0, 255),
            )

            return frame

        blur_score = (
            BlurDetector.get_blur_score(
                frame
            )
        )

        if blur_score < self.blur_threshold:

            self.stable_frame_count = 0

            self.draw_text(
                frame,
                "IMAGE TOO BLURRY",
                150,
                (0, 0, 255),
            )

            return frame

        brightness = (
            BrightnessChecker.get_brightness(
                frame
            )
        )

        if (
            brightness < self.brightness_min
            or brightness > self.brightness_max
        ):

            self.stable_frame_count = 0

            self.draw_text(
                frame,
                "BAD LIGHTING",
                150,
                (0, 0, 255),
            )

            return frame

        valid_pose = (
            PoseValidator.validate_pose(
                face,
                current_pose,
            )
        )

        if not valid_pose:

            self.stable_frame_count = 0

            self.draw_text(
                frame,
                (
                    f"PLEASE LOOK "
                    f"{current_pose.upper()}"
                ),
                150,
                (0, 0, 255),
            )

            return frame

        self.stable_frame_count += 1

        self.draw_text(
            frame,
            (
                f"STABLE: "
                f"{self.stable_frame_count}/"
                f"{self.stable_frames_required}"
            ),
            150,
        )

        if (
            self.stable_frame_count
            < self.stable_frames_required
        ):

            return frame

        current_time = time.time()

        if (
            current_time - self.last_capture_time
            >= self.capture_interval
        ):

            self.save_image(
                frame,
                current_pose,
            )

            self.current_pose_count += 1

            self.last_capture_time = current_time

            self.stable_frame_count = 0

            self.window.progress_panel.update_progress(
                current_pose,
                self.current_pose_count,
                self.capture_per_pose,
            )

        if (
            self.current_pose_count
            >= self.capture_per_pose
        ):

            self.current_pose_index += 1

            self.current_pose_count = 0

            self.stable_frame_count = 0

        return frame

    def update_loop(self):

        ret, frame = self.cap.read()

        if ret:

            frame = cv2.flip(
                frame,
                1,
            )

            frame = self.process_frame(
                frame
            )

            self.window.webcam_panel.update_frame(
                frame
            )

        self.window.after(
            10,
            self.update_loop,
        )

    def run(self):

        self.update_loop()

        self.window.mainloop()

        self.cap.release()

if __name__ == "__main__":

    auto_capture_dataset = (
        AutoCaptureDataset()
    )

    auto_capture_dataset.run()