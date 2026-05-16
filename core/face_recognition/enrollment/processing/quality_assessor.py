
import cv2
import numpy

class QualityAssessor:
    @staticmethod
    def blur_score(face_crop):

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    @staticmethod
    def brightness_score(face_crop):
        
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        return numpy.mean(gray)
    
    @staticmethod
    def evaluate(face_crop):

        blur = QualityAssessor.blur_score(face_crop)
        brightness = QualityAssessor.brightness_score(face_crop)

        blur_is_ok = blur > 100
        brightness_is_ok = 70 < brightness < 180

        return {
            "blur": blur,
            "brightness": brightness,
            "blur_is_ok": blur_is_ok,
            "brightness_is_ok": brightness_is_ok,
            "is_valid": blur_is_ok and brightness_is_ok,
        }


