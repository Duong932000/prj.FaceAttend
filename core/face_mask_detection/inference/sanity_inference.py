#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      facemask_inference.py
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


import torch
import cv2
import numpy as np
from pathlib import Path
import torch.nn.functional as F

from models.MobileNetV3 import get_model


CLASS_NAMES = [
    'mask',
    'mask_chin',
    'mask_mouth_chin',
    'mask_nose_mouth',
    'no_mask'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weight_path):
    """
    Load the pre-trained model from the specified path.
    """

    model = get_model(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess(img):
    """
    Preprocess the input image for inference.
    """

    if img is None:
        raise ValueError("Image not found")

    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    return torch.tensor(img, dtype=torch.float32).to(device)

def predict(model, img_path):
    """
    Perform inference on a single image.
    """

    img = cv2.imread(str(img_path))
    input_tensor = preprocess(img)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    return {
        "label": CLASS_NAMES[pred.item()],
        "class_id": pred.item(),
        "confidence": round(confidence.item(), 4)
    }

def test_samples(model, dataset_test_dir):
    """
    Test the model on a set of sample images.
    """
    
    results = []
    correct = 0

    for cls in CLASS_NAMES:
        img_path = Path(dataset_test_dir) / f"{cls}.jpg"

        if not img_path.exists():
            print(f"Missing: {img_path}")
            continue

        pred = predict(model, img_path)

        is_correct = pred["label"] == cls
        if is_correct:
            correct += 1

        results.append({
            "GT": cls,
            "Pred": pred["label"],
            "Conf": pred["confidence"],
            "OK": "✔" if is_correct else "✘"
        })

    acc = correct / len(results) if results else 0

    return results, acc

def print_table(results, acc):

    print("\n================= Face Mask Inference Test =================")
    print(f"{'GT':20} {'Pred':20} {'Conf':10} {'OK'}")
    print("-" * 60)

    for r in results:
        print(f"{r['GT']:20} {r['Pred']:20} {r['Conf']:<10} {r['OK']}")

    print("-" * 60)
    print(f"Accuracy: {acc:.2f}")

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parents[1]

    weight_path = BASE_DIR / "weights" / "best_mask_model.pth"
    dataset_test_dir = BASE_DIR / "datasets"

    model = load_model(weight_path)
    results, acc = test_samples(model, dataset_test_dir)

    print_table(results, acc)
