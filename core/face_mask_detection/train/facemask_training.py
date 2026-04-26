#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      facemask-training.py
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
import yaml
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from dataloaders.facemask_dataloader import get_dataloaders
from models.MobileNetV3 import get_model

ROOT_DIR = Path(__file__).resolve().parents[3]

def evaluate(model, dataloader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0

def evaluate_detailed(model, dataloader, device, class_names):

    model.eval()

    num_classes = len(class_names)
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0

    return acc, conf_matrix

def display_evaluation_report(conf_matrix, class_names, acc):

    print("\n=== Evaluation Report ===")
    print(f"{'Class':20} {'Precision':10} {'Recall':10} {'Support':10}")
    print("-" * 60)

    for i in range(len(class_names)):
        TP = conf_matrix[i, i].item()
        FP = conf_matrix[:, i].sum().item() - TP
        FN = conf_matrix[i, :].sum().item() - TP
        support = conf_matrix[i, :].sum().item()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        print(f"{class_names[i]:20} {precision:<10.4f} {recall:<10.4f} {support:<10}")

    print("-" * 60)
    print(f"Overall Accuracy: {acc:.4f}")

def plot_training(history, save_path):

    epochs = range(1, len(history["loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["loss"], label="Train Loss")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curve")
    plt.legend()

    plt.savefig(save_path)
    plt.close()

    print(f"Training chart saved at: {save_path}")

def train(config_path):

    # Load config.yml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = get_dataloaders(config)

    print("Class mapping:", train_loader.dataset.classes)

    # Get Model
    model = get_model(config["num_classes"], config["model"]["pretrained"])
    model.to(device)

    # Loss and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    epochs = config["epochs"]
    best_acc = 0.0

    # Output path
    model_output_dir = ROOT_DIR / config["model_output_dir"]
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model_output_path = model_output_dir / config["face_mask_model_name"]

    history = {
        "loss": [],
        "val_acc": []
    }
    
    # training loop
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

    print("\nTraining completed!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    class_names = train_loader.dataset.classes
    acc, conf_matrix = evaluate_detailed(model, val_loader, device, class_names)
    display_evaluation_report(conf_matrix, class_names, acc)

    # Save chart
    chart_path = model_output_dir / "training_curve.png"
    plot_training(history, chart_path)

if __name__ == "__main__":

    CONFIG_PATH = ROOT_DIR / "core" / "face_mask_detection" / "configs" / "config.yml"

    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}")
        exit(1)

    train(CONFIG_PATH)
