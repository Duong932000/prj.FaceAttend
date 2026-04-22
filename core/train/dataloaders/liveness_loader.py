#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      liveness-loader.py
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


# import necessary libraries
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),

        # Anti-spoof augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomGrayscale(p=0.1),

        transforms.ToTensor(),

        # normalize [-1, 1]
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def create_dataset(data_dir, train=True):
    transform = get_train_transforms() if train else get_val_transforms()
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    return dataset

def create_weighted_sampler(dataset):
    class_counts = [0] * len(dataset.classes)

    for _, label in dataset.samples:
        class_counts[label] += 1

    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)

    sample_weights = []
    for _, label in dataset.samples:
        sample_weights.append(class_weights[label])

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

def get_dataloaders(data_dir, batch_size=32, train=True, use_weighted_sampler=True, num_workers=4):

    dataset = create_dataset(data_dir, train=train)

    if train and use_weighted_sampler:
        sampler = create_weighted_sampler(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)

    return loader

def inspect_dataset(data_dir):
    dataset = datasets.ImageFolder(data_dir)

    print("Classes:", dataset.classes)
    print("Total samples:", len(dataset))

    class_counts = {cls: 0 for cls in dataset.classes}
    for _, label in dataset.samples:
        class_counts[dataset.classes[label]] += 1

    print("Class distribution:")
    for k, v in class_counts.items():
        print(f"  {k}: {v}")
