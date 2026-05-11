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
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from core.face_mask_detection.models.MobileNetV3 import get_model
from core.face_mask_detection.evaluation.plot import plot_training_curve
from core.face_mask_detection.dataloaders.facemask_dataloader import get_dataloaders
from core.face_mask_detection.evaluation.eval import evaluate, evaluate_detailed, display_evaluation_report

ROOT_DIR = Path(__file__).resolve().parents[3]

class EarlyStopping:
    """Production early stopping"""
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, val_score: float, model: torch.nn.Module):
        score = -val_score  # Maximize acc
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered!")
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        if self.restore_best:
            self.best_weights = model.state_dict().copy()

def train(config_path: Path):
    start_time = time.time()
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}")
    
    # Data
    train_loader, val_loader = get_dataloaders(config)
    class_names = train_loader.dataset.classes
    
    # Model - FULLY CONFIG DRIVEN
    model = get_model(
        num_classes=config["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout_rate=float(config["dropout_rate"]),           # ← FLOAT!
        freeze_backbone=config["freeze_backbone"],
        width_mult=float(config["width_mult"])                # ← FLOAT!
    )
    model.to(device)
    
    # Loss + Optimizer - SAFE FLOAT CONVERSION
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["label_smoothing"])
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config["learning_rate"]),                    # ← FLOAT!
        weight_decay=float(config["weight_decay"])            # ← FIX: str → float!
    )
    
    # Scheduler - FULL CONFIG
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=int(config["scheduler_patience"]),
        factor=float(config["scheduler_factor"]),
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=int(config["early_stopping_patience"]))
    
    epochs = int(config["epochs"])
    history = {"train_loss": [], "val_acc": [], "lr": []}
    
    # Paths
    model_pth_out_dir = ROOT_DIR / config["model_pth_out_dir"]
    model_pth_out_dir.mkdir(parents=True, exist_ok=True)
    model_output_path = model_pth_out_dir / config["face_mask_pth_name"]
    
    print("🔄 Training started...")
    
    best_acc = 0.0
    for epoch in range(epochs):
        # === TRAIN ===
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # === VALIDATE ===
        val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # History
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")
        
        # === EARLY STOPPING ===
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            if early_stopping.restore_best:
                model.load_state_dict(early_stopping.best_weights)
                print("🔄 Restored best weights")
            break
        
        # === SAVE BEST ===
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_output_path)
            print(f"--> New best model: {val_acc:.4f}")
    
    # === FINAL EVALUATION ===
    final_acc, conf_matrix = evaluate_detailed(model, val_loader, device, class_names)
    display_evaluation_report(conf_matrix, class_names, final_acc)
    
    # === PLOT + EXPORT ===
    chart_path = model_pth_out_dir / "training_curve.png"
    plot_training_curve(history, chart_path)
    
    # Model info
    end_time = time.time()
    model_info = {
        "model_name": config["face_mask_pth_name"],
        "model_type": "MobileNetV3-Large",
        "num_classes": config["num_classes"],
        "classes": class_names,
        "training": {
            "epochs_trained": epoch + 1,
            "learning_rate": float(config["learning_rate"]),
            "batch_size": config["batch_size"],
            "weight_decay": float(config["weight_decay"]),
            "label_smoothing": float(config["label_smoothing"]),
            "device": str(device),
        },
        "dataset": {
            "path": config["dataset_path"],
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
        },
        "performance": {
            "best_val_accuracy": float(best_acc),
            "final_accuracy": float(final_acc),
        },
        "time": {
            "start": datetime.fromtimestamp(start_time).isoformat(),
            "end": datetime.fromtimestamp(end_time).isoformat(),
            "duration_sec": round(end_time - start_time, 2)
        },
        "author": "Duong",
        "config_hash": str(hash(str(config)))
    }
    
    json_path = model_pth_out_dir / "model_info.json"
    json_path.write_text(json.dumps(model_info, indent=2))
    
    print(f"✅ Training completed!")
    print(f"📁 Model saved: {model_output_path}")
    print(f"📊 Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":

    CONFIG_PATH = ROOT_DIR / "core" / "face_mask_detection" / "configs" / "config.yml"
    train(CONFIG_PATH)
