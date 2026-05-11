#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      facemask-dataloader.py
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRAIN_DIR = "train"
VAL_DIR = "val"

def get_dataloaders(config):

    input_size = config["input_size"]
    dataset_path = os.path.expanduser(config["dataset_path"])
    batch_size = config["batch_size"]

    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path not found: {dataset_path}")

    train_transform = transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32)),  # Random crop source
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.get("mean", [0.485, 0.456, 0.406]), 
            std=config.get("std", [0.229, 0.224, 0.225])
        )
    ])

    # Clean validation
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.get("mean", [0.485, 0.456, 0.406]), 
            std=config.get("std", [0.229, 0.224, 0.225])
        )
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, TRAIN_DIR),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, VAL_DIR),
        transform=val_transform
    )

    print(f"Dataset: train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    num_workers = min(8, os.cpu_count() or 4)
    pin_memory = config["device"] == "cuda"
    print(f"Num worker: {num_workers}")
    print(f"Device: {pin_memory}")

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              drop_last=True
                              )

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory
                            )

    return train_loader, val_loader
