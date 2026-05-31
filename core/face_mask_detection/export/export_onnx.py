import torch

from pathlib import Path

from core.face_mask_detection.models.MobileNetV3 import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_onnx(model_path, output_path):

    model = get_model(
        num_classes=5,
        pretrained=False,
        dropout_rate=0.3,
        freeze_backbone=False,
        width_mult=1.0
    )

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.eval()

    dummy_input = torch.randn(1, 3, 112, 112)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=17
    )

    print(f"ONNX exported: {output_path}")


if __name__ == "__main__":

    root_dir = Path(__file__).resolve().parents[3]

    model_path = (
        root_dir
        / "output"
        / "face_mask_detection"
        / "pth"
        / "face_mask_model.pth"
    )

    output_path = (
        root_dir
        / "output"
        / "face_mask_detection"
        / "onnx"
        / "face_mask_model.onnx"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_onnx(model_path, output_path)