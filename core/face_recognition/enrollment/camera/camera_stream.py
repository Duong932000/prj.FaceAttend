#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      camera_stream.py
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
import threading


class CameraStream:

    def __init__(self, camera_cfg):

        # =====================================================
        # CAMERA CONFIG
        # =====================================================
        self.camera_id = camera_cfg["camera_id"]
        self.width = camera_cfg["width"]
        self.height = camera_cfg["height"]

        # =====================================================
        # VIDEO CAPTURE
        # =====================================================
        self.cap = cv2.VideoCapture(self.camera_id)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Optional
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # =====================================================
        # THREAD CONTROL
        # =====================================================
        self.running = False
        self.thread = None

        # =====================================================
        # FRAME STORAGE
        # =====================================================
        self.frame = None
        self.lock = threading.Lock()

    # =========================================================
    # START CAMERA
    # =========================================================
    def start(self):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        self.thread.start()

    # =========================================================
    # UPDATE FRAME LOOP
    # =========================================================
    def update(self):

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                continue

            # mirror webcam
            frame = cv2.flip(frame, 1)

            with self.lock:
                self.frame = frame

    # =========================================================
    # GET LATEST FRAME
    # =========================================================
    def get_latest_frame(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()

    # =========================================================
    # STOP CAMERA
    # =========================================================
    def stop(self):

        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1)

        if self.cap.isOpened():
            self.cap.release()
