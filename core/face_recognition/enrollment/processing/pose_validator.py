
import numpy

class PoseValidator:
    @staticmethod
    def __init__(face, target_crop):

        yaw, pitch, _ = face.pose

        if target_crop == ""