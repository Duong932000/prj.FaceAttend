
import cv2
from core.face_recognition.utils.config import load_enrollment_config
from core.face_recognition.enrollment.ui.main_window import MainWindow
from core.face_recognition.enrollment.camera.camera_stream import CameraStream
from core.face_recognition.enrollment.processing.face_processor import FaceProcessor

class FaceEnrollmentApp:
    def __init__(self):
        _, self.cfg = load_enrollment_config()

        self.window = MainWindow()
        self.camera_stream = CameraStream(self.cfg["enrollment"]["camera"])
        self.processor = FaceProcessor(self.camera_stream)
        self.camera_stream.start()
        self.processor.start()

    def draw_overlay(self, frame, result):
        if not result["face_detected"]:
            cv2.putText(frame, "NO FACE DETECTED", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        face = result["face"]

        x1, y1, x2, y2 = face.bbox.astype(int)

        color = (0, 255, 255)

        if (result["stable"] and result["pose_valid"]):
            color = (0, 255, 0)

        if not result["pose_valid"]:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        return frame

    def update_ui(self, result):

        status = "ALIGNING"

        if (result["stable"] and result["pose_valid"]):
            status = "READY"

        self.window.right_panel.status_label.configure(text=status)
        self.window.right_panel.pose_label.configure(text=(f"POSE: {self.processor.target_pose.upper()}"))

    def render_loop(self):

        result = self.processor.latest_result
        if result is not None:
            frame = result["frame"]
            frame = self.draw_overlay(frame, result)
            self.update_ui(result)
            self.window.webcam_panel.update_frame(frame)

        self.window.after(30, self.render_loop)

    def run(self):
        self.render_loop()
        self.window.mainloop()
        self.camera_stream.stop()

if __name__ == "__main__":

    app = FaceEnrollmentApp()
    app.run()