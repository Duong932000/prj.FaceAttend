#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      stability_tracker.py
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


import numpy as np
from collections import deque

class StabilityTracker:
    def __init__(self, stability_cfg):

        self.history_size = stability_cfg["history_size"]

        self.center_std_max = stability_cfg["center_std_max"]
        self.center_std_y_max = stability_cfg["center_std_y_max"]

        self.yaw_std_max = stability_cfg["yaw_std_max"]
        self.pitch_std_max = stability_cfg["pitch_std_max"]

        self.center_history = deque(maxlen=self.history_size)
        self.yaw_history = deque(maxlen=self.history_size)
        self.pitch_history = deque(maxlen=self.history_size)

    def update(self, face):

        x1, y1, x2, y2 = face.bbox.astype(int)

        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)

        yaw, pitch, _ = face.pose
        self.center_history.append([cx, cy])
        self.yaw_history.append(float(yaw))
        self.pitch_history.append(float(pitch))

    def reset(self):

        self.center_history.clear()

        self.yaw_history.clear()

        self.pitch_history.clear()

    def is_stable(self):

        if len(self.center_history) < self.history_size:
            return False

        centers = np.array(list(self.center_history), dtype=np.float32).reshape(-1, 2)

        center_std_x = np.std(centers[:, 0])
        center_std_y = np.std(centers[:, 1])

        yaw_std = np.std(np.array(self.yaw_history, dtype=np.float32))

        pitch_std = np.std(np.array(self.pitch_history, dtype=np.float32))

        return (
            center_std_x < self.center_std_max
            and
            center_std_y < self.center_std_y_max
            and
            yaw_std < self.yaw_std_max
            and
            pitch_std < self.pitch_std_max
        )