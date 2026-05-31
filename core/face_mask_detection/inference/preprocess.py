import cv2
import numpy as np
import torch


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FacePreprocessor:

    def __init__(self, image_size=112):
        self.image_size = image_size

    def preprocess(self, face_crop, device):

        image = cv2.resize(face_crop, (self.image_size, self.image_size))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0

        image = (image - MEAN) / STD

        image = np.transpose(image, (2, 0, 1))

        image = np.expand_dims(image, axis=0)

        tensor = torch.tensor(image, dtype=torch.float32)

        return tensor.to(device)