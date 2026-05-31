import torch
import torch.nn.functional as F


CLASS_NAMES = [
    "mask",
    "mask_chin",
    "mask_mouth_chin",
    "mask_nose_mouth",
    "no_mask"
]


class MaskClassifier:

    def __init__(self, model, preprocessor, device):

        self.model = model
        self.preprocessor = preprocessor
        self.device = device

    def predict(self, face_crop):

        input_tensor = self.preprocessor.preprocess(
            face_crop,
            self.device
        )

        with torch.no_grad():

            logits = self.model(input_tensor)

            probs = F.softmax(logits, dim=1)

            confidence, pred = torch.max(probs, dim=1)

        class_name = CLASS_NAMES[pred.item()]

        return {
            "class_name": class_name,
            "confidence": float(confidence.item()),
            "class_id": int(pred.item())
        }