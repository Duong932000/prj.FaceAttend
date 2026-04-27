#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      test_export_onnx.py
# .:ccccccccccccc;KMMc;cc;xMMc;ccccccc:.    Description  :      Export trained model to ONNX format
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
import onnxruntime
import numpy
import cv2
from pathlib import Path
import torch.nn.functional as F

from models.MobileNetV3 import get_model

CLASS_NAMES = [
    "mask",
    "mask_chin",
    "mask_mouth_chin",
    "mask_nose_mouth",
    "no_mask"
]

INPUT_SIZE = 112

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = Path(__file__).resolve().parents[3]

MODEL_OUT_DIR = ROOT_DIR / "output" / "face_mask_detection"
PTH_PATH = MODEL_OUT_DIR / "pth" / "face_mask_model.pth"
ONNX_PATH = MODEL_OUT_DIR / "onnx" / "face_mask_model.onnx"

TEST_IMAGE = ROOT_DIR/ "core" / "face_mask_detection" / "datasets" / "mask.jpg"

def preprocess(img):
    if img is None:
        raise ValueError("Input image is None")
    
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0

    mean = numpy.array([0.485, 0.456, 0.406])
    std = numpy.array([0.229, 0.224, 0.225])
    
    img = (img - mean) / std

    img = numpy.transpose(img, (2, 0, 1))
    img = numpy.expand_dims(img, 0).astype(numpy.float32)

    return img

def load_pytorch():
    model = get_model(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(PTH_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model

def load_onnx():

    session = onnxruntime.InferenceSession(str(ONNX_PATH))

    return session

def inference_pytorch(model, input_np):

    input_tensor = torch.tensor(input_np).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)

    return probs.cpu().numpy()

def inference_onnx(session, input_np):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_np})
    probs = outputs[0]
    probs = softmax_numpy(probs)
    return probs

def softmax_numpy(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def compare_outputs(pt_out, onnx_out):

    diff = numpy.abs(pt_out - onnx_out)
    max_diff = diff.max()

    print("\n=== Comparison ===")
    print(f"Max difference: {max_diff:.6f}")

    if max_diff < 1e-4:
        print("✔ ONNX matches PyTorch")
    else:
        print("⚠ Warning: difference is large")

def display_results(probs, tag):

    pred = numpy.argmax(probs)
    conf = probs[0][pred]

    print(f"\n[{tag}]")
    print(f"Label: {CLASS_NAMES[pred]}")
    print(f"Confidence: {conf:.4f}")
    print(f"Raw probs: {numpy.round(probs, 4)}")

def main():

    print("Loading models...")
    pt_model = load_pytorch()
    onnx_model = load_onnx()

    print("Loading image...")
    img = cv2.imread(str(TEST_IMAGE))
    input_np = preprocess(img)

    print("Running PyTorch inference...")
    pt_out = inference_pytorch(pt_model, input_np)

    print("Running ONNX inference...")
    onnx_out = inference_onnx(onnx_model, input_np)

    display_results(pt_out, "PyTorch")
    display_results(onnx_out, "ONNX")

    compare_outputs(pt_out, onnx_out)

if __name__ == "__main__":
    main()