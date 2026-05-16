
import numpy
from collections import deque


class StabilityTracker:
    def __init__(self):

        self.center_history = deque(maxlen=10)
        self.yaw_history = deque(maxlen=10)
        self.pitch_history = deque(maxlen=10)

    def update(self, face):
        x1, y1, x2, y2 = face.bbox.astype(int)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        yaw, pitch, _ = face.pose

        self.center_history.append((cx, cy))
        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)

    def is_stable(self):

        if len(self.center_history) < 10:
            return False
        
        centers = numpy.array(self.center_history)
        center_std = numpy.std(centers, axis=0)
        yaw_std = numpy.std(self.yaw_history)
        pitch_std = numpy.std(self.pitch_history)

        return (center_std[0] < 12 and center_std[1] < 12 and yaw_std < 3 and pitch_std < 3)

