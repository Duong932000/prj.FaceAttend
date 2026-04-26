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
from pathlib import Path

from dataloaders.facemask_dataloader import get_dataloaders
from models.MobileNetV3 import get_model


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

def train(config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(config)

    # Debug class order
    print("Class mapping:", train_loader.dataset.classes)
    
    # get model
    model = get_model(config["num_classes"], config["model"]["pretrained"])
    model.to(device)

    # config loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    epochs = config["epochs"]
    best_acc = 0.0

    os.makedirs("weights", exist_ok=True)
    
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

        # validation accuracy
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "weights/best_mask_model.pth")
            print(f"New best model saved (Accuracy: {best_acc:.4f})")

    print("Training completed!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parents[1]
    CONFIG_PATH = BASE_DIR / "configs" / "config.yml"

    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}")
        exit(1)

    train(CONFIG_PATH)
