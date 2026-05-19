#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      quality_assessor.py
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
import numpy

class QualityAssessor:
    def __init__(self, quality_cfg):
        self.blur_threshold = quality_cfg["blur_threshold"]
        self.brightness_min = quality_cfg["brightness_min"]
        self.brightness_max = quality_cfg["brightness_max"]

    def blur_score(self, face_crop):

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def brightness_score(self, face_crop):

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        return numpy.mean(gray)

    def evaluate(self, face_crop):

        blur = self.blur_score(face_crop)

        brightness = self.brightness_score(face_crop)

        blur_ok = blur > self.blur_threshold

        brightness_ok = self.brightness_min < brightness < self.brightness_max

        return {
            "blur": blur,
            "brightness": brightness,
            "blur_ok": blur_ok,
            "brightness_ok": brightness_ok,
            "valid": (blur_ok and brightness_ok)
        }
