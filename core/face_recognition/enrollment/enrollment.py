#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      enrollment.py
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
from core.face_recognition.utils.config import load_enrollment_config
from core.face_recognition.enrollment.ui.main_window import MainWindow
from core.face_recognition.enrollment.camera.camera_stream import CameraStream
from core.face_recognition.enrollment.processing.face_processor import FaceProcessor

class FaceEnrollmentApp:
    def __init__(self):
        _, self.cfg = load_enrollment_config()

        self.person_name = None
        self.enrollment_started = False
        
        # show window
        self.window = MainWindow(start_detection_callback=self.start_detection,
                                 stop_detection_callback=self.stop_detection,
                                 start_enrollment_callback=self.start_enrollment)

        # stream camera
        self.camera_stream = CameraStream(self.cfg["enrollment"]["camera"])
        self.camera_stream.start()

        self.processor = FaceProcessor(self.camera_stream)
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

    def start_detection(self):

        self.processor.enable_detection()

        self.window.right_panel.status_label.configure(text="DETECTION ON")

    def stop_detection(self):

        self.processor.disable_detection()

        self.window.right_panel.status_label.configure(text="DETECTION OFF")

    def start_enrollment(self, person_name):

        self.person_name = person_name

        self.enrollment_started = True

        print(f"[INFO] Enrollment started: {person_name}")

    def render_loop(self):

        result = self.processor.latest_result

        frame = self.camera_stream.get_latest_frame()

        if frame is None:
            self.window.after(30, self.render_loop)
            return
        
        if (self.processor.detection_enabled and result is not None):
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