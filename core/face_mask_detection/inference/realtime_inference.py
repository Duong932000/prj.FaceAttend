#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      realtime_inference.py
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


import os
import cv2
import torch
import numpy
import torch.nn.functional as F
from pathlib import Path

from core.face_mask_detection.models.MobileNetV3 import get_model

CLASS_NAMES = [
    'mask',
    'mask_chin',
    'mask_mouth_chin',
    'mask_nose_mouth',
    'no_mask'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_output_path):

    model = get_model(num_classes=len(CLASS_NAMES), pretrained=True)
    model.load_state_dict(torch.load(model_output_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def preprocess(img):

    if img is None:
        return None
    
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img / 255.0
    mean = numpy.array([0.485, 0.456, 0.406])
    std  = numpy.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    img = numpy.transpose(img, (2, 0, 1))
    img = numpy.expand_dims(img, 0)
    
    return torch.tensor(img, dtype=torch.float32).to(device)

def predict(model, frame):

    input_tensor = preprocess(frame)
    if input_tensor is None:
        return None
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    return {
        "label": CLASS_NAMES[pred.item()],
        "class_id": pred.item(),
        "confidence": confidence.item()
    }

def main():

    # portable OpenCV GUI fix
    os.environ["QT_QPA_PLATFORM"] = "xcb"  # Fedora+Ubuntu
    os.environ["DISPLAY"] = ":0"           # Server fallback

    model_name = "face_mask_model.pth"
    ROOT_DIR = Path(__file__).resolve().parents[3]

    # get model output path
    model_output_path = ROOT_DIR / "output" / "face_mask_detection" / "pth" / model_name

    print(f"[INFO] Loading model from: {model_output_path}")
    if not model_output_path.exists():
        print(f"[ERROR] Model not found at {model_output_path}")
        return
    
    model = load_model(model_output_path)
    print(f"[INFO] Model loaded on {device}")

    # open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam")
        return
    
    print("[INFO] Press 'q' to quit ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            break
        
        # Inference
        result = predict(model, frame)
        
        # Draw results
        if result:
            label = result["label"]
            conf = result["confidence"]
            color = (0, 255, 0) if conf > 0.9 else (0, 165, 255)
            
            cv2.rectangle(frame, (10, 10), (400, 80), color, -1)
            cv2.putText(frame, f"{label}: {conf:.2f}", (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Show frame
        cv2.imshow('Face Mask Detection - Realtime', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed")

if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        raise(e)