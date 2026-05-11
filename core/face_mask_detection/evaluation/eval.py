#########################################################
#             .',;::::;,'.                 
#          .';:cccccccccccc:;,.              
#       .;cccccccccccccccccccccc;           --------------
#     .:cccccccccccccccccccccccccc:.        Project name :      prj.FaceAttend
#   .;ccccccccccccc;.:dddl:.;ccccccc;.      Author       :      Nguyen Dac Duong
#  .:ccccccccccccc;OWMKOOXMWd;ccccccc:.     File name    :      eval.py
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


import torch
import numpy as np


def evaluate(model: torch.nn.Module, dataloader, device: torch.device) -> float:
    """Simple accuracy"""

    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

def evaluate_detailed(model: torch.nn.Module, dataloader, device: torch.device, 
                     class_names: list) -> tuple[float, torch.Tensor]:
    """Full metrics + confusion matrix"""

    model.eval()
    num_classes = len(class_names)
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32, device=device)
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_matrix[t, p] += 1
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = correct / total

    return acc, conf_matrix.cpu()

def display_evaluation_report(conf_matrix: torch.Tensor, class_names: list, acc: float):
    """Pretty metrics table"""

    print("\n=== EVALUATION REPORT ===")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"{'Class':<15} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Supp':<8}")
    print("-" * 55)
    
    for i, cls in enumerate(class_names):
        TP = conf_matrix[i, i].item()
        FP = conf_matrix[:, i].sum().item() - TP
        FN = conf_matrix[i, :].sum().item() - TP
        support = conf_matrix[i, :].sum().item()
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{cls:<15} {precision:<7.3f} {recall:<7.3f} {f1:<7.3f} {support:<7}")
    
    print("-" * 55)
