#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      export_onnx.py
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


import os
import yaml
import torch
from pathlib import Path

from models.MobileNetV3 import get_model


ROOT_DIR = Path(__file__).resolve().parents[3]

def load_config():

    config_path = ROOT_DIR / "core" / "face_mask_detection" / "configs" / "config.yml"
    if not config_path.exists():
        print(f"Config file not found at {config_path}")
        return None
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def export_to_onnx():

    config = load_config()

    device = torch.device("cpu")

    model_onnx_dir = ROOT_DIR / config["model_onnx_out_dir"]
    os.makedirs(model_onnx_dir, exist_ok=True)

    pth_path = ROOT_DIR / config["model_pth_out_dir"] / config["face_mask_pth_name"]

    onnx_path = model_onnx_dir / config["face_mask_onnx_name"]

    if not pth_path.exists():
        print(f"Model file not found at {pth_path}")
        return

    print(f"Loading model from: {pth_path}")

    # load model
    model = get_model(config["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()

    # dummy input for ONNX export
    input_size = config["input_size"]
    dummy_input = torch.randn(1 ,3, input_size, input_size).to(device)

    # export ONNX
    print(f"Exporting model to ONNX format at: {onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,        # stable opset for most use cases
        do_constant_folding=True,

        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print("ONNX export completed successfully.")
    print(f"ONNX model saved at: {onnx_path}")

if __name__ == "__main__":
    try:
        export_to_onnx()
    except Exception as e:
        print(f"Error during ONNX export: {e}")