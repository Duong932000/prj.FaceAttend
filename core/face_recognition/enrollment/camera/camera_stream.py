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

        self.cap = cv2.VideoCapture(camera_cfg["camera_id"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["height"])

        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):

        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            with self.lock:
                self.frame = frame

    def get_latest_frame(self):

        with self.lock:
            if self.frame is None:
                return None
            
            return self.frame.copy()

    def stop(self):

        self.running = False
        self.cap.release()
