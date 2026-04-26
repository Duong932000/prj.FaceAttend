#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      liveness-training.py
# .:ccccccccccccc;KMMc;cc;xMMc;ccccccc:.    Description  :      Train a liveness detection model using the LivenessNet architecture.
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


# import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim

# import custom modules
from models.liveness.minifasnet import MiniFASNet
from dataloaders.liveness_loader import get_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_LIVENESS_DIR = "core/train/datasets/liveness"

train_loader = get_dataloaders(f"{DATASET_LIVENESS_DIR}/train", batch_size=64, train=True)

val_loader = get_dataloaders(f"{DATASET_LIVENESS_DIR}/val", batch_size=64, train=False)

model = MiniFASNet(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

EPOCHS = 20

# training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    scheduler.step()
    train_acc = correct / total

    # validation phase
    model.eval()
    val_corrcet = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(
        f"[Epoch {epoch+1}] "
        f"Loss: {total_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )


# save model
SAVED_MODEL_NAME = "liveness_minifasnet.pth"
torch.save(model.state_dict(), SAVED_MODEL_NAME)
print(f"Model saved to {SAVED_MODEL_NAME}")