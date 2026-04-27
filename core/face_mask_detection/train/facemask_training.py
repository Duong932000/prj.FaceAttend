#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      facemask-training.py
# .:ccccccccccccc;KMMc;cc;xMMc;ccccccc:.    Description  :      Train face mask detection model
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


import json
import time
import yaml
import torch
from pathlib import Path
from datetime import datetime

from models.MobileNetV3 import get_model
from evaluation.plot import plot_training_curve
from dataloaders.facemask_dataloader import get_dataloaders
from evaluation.eval import evaluate, evaluate_detailed, display_evaluation_report

ROOT_DIR = Path(__file__).resolve().parents[3]

def train(config_path):

    start_time = time.time()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # get data loader
    train_loader, val_loader = get_dataloaders(config)

    # get class names
    class_names = train_loader.dataset.classes
    print("Class mapping:", class_names)

    # get model
    model = get_model(config["num_classes"], config["model"]["pretrained"])
    model.to(device)

    # training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    epochs = config["epochs"]
    best_acc = 0.0

    model_pth_out_dir = ROOT_DIR / config["model_pth_out_dir"]
    model_pth_out_dir.mkdir(parents=True, exist_ok=True)

    model_output_path = model_pth_out_dir / config["face_mask_pth_name"]

    history = {
        "loss": [],
        "val_acc": []
    }

    # Training loop
    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)

        history["loss"].append(total_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_output_path)
            print(f"New best model saved (Accuracy: {best_acc:.4f})")

    # Final evaluation
    acc, conf_matrix = evaluate_detailed(model, val_loader, device, class_names)
    display_evaluation_report(conf_matrix, class_names, acc)

    # Plot training curve
    chart_path = model_pth_out_dir / "training_curve.png"
    plot_training_curve(history, chart_path)

    # Export model info
    end_time = time.time()
    model_info = {
        "model_name": config["face_mask_pth_name"],
        "model_type": "MobileNetV3",
        "num_classes": config["num_classes"],
        "classes": class_names,

        "training": {
            "epochs": epochs,
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "device": str(device),
        },

        "dataset": {
            "path": config["dataset_path"],
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
        },

        "performance": {
            "best_val_accuracy": round(best_acc, 4),
            "final_accuracy": round(acc, 4),
        },

        "time": {
            "start": datetime.fromtimestamp(start_time).isoformat(),
            "end": datetime.fromtimestamp(end_time).isoformat(),
            "duration_sec": round(end_time - start_time, 2)
        },

        "author": "Duong",
        "created_at": datetime.now().isoformat()
    }

    json_path = model_pth_out_dir / "model_info.json"
    with open(json_path, "w") as f:
        json.dump(model_info, f, indent=4)

    print(f"\nModel info saved at: {json_path}")
    print("Training completed!")

if __name__ == "__main__":

    CONFIG_PATH = ROOT_DIR / "core" / "face_mask_detection" / "configs" / "config.yml"

    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}")
        exit(1)

    train(CONFIG_PATH)
