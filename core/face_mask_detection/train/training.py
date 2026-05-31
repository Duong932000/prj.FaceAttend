import json
import time
import yaml
import torch
import torch.nn as nn

from pathlib import Path
from datetime import datetime

from core.face_mask_detection_OLD.models.MobileNetV3 import get_model
from core.face_mask_detection_OLD.dataloaders.facemask_dataloader import get_dataloaders
from core.face_mask_detection_OLD.evaluation.eval import (
    evaluate,
    evaluate_detailed
)
from core.face_mask_detection_OLD.evaluation.plot import plot_training_curve


ROOT_DIR = Path(__file__).resolve().parents[3]


# =========================================================
# EARLY STOPPING
# =========================================================

class EarlyStopping:

    def __init__(
        self,
        patience=7,
        min_delta=0.001,
        restore_best=True
    ):

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_score, model):

        if self.best_score is None:

            self.best_score = val_score

            if self.restore_best:
                self.best_weights = model.state_dict().copy()

            return

        if val_score > self.best_score + self.min_delta:

            self.best_score = val_score
            self.counter = 0

            if self.restore_best:
                self.best_weights = model.state_dict().copy()

        else:

            self.counter += 1

            print(
                f"EarlyStopping counter: "
                f"{self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:

                self.early_stop = True

                print("Early stopping triggered")


# =========================================================
# TRAIN
# =========================================================

def train(config_path):

    start_time = time.time()

    # -----------------------------------------------------
    # CONFIG
    # -----------------------------------------------------

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        config["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"🚀 Training on {device}")

    # -----------------------------------------------------
    # DATALOADER
    # -----------------------------------------------------

    train_loader, val_loader = get_dataloaders(config)

    class_names = train_loader.dataset.classes

    print(
        f"Dataset: "
        f"train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}"
    )

    print(f"Classes: {class_names}")

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = get_model(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout_rate=float(config["model"]["dropout_rate"]),
        freeze_backbone=config["model"]["freeze_backbone"],
        width_mult=float(config["model"]["width_mult"])
    )

    model.to(device)

    print("✅ Model created")

    # -----------------------------------------------------
    # LOSS
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(
            config["training"]["label_smoothing"]
        )
    )

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(
            config["training"]["weight_decay"]
        )
    )

    # -----------------------------------------------------
    # SCHEDULER
    # -----------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=3,
        factor=0.5
    )

    # -----------------------------------------------------
    # EARLY STOPPING
    # -----------------------------------------------------

    early_stopping = EarlyStopping(
        patience=7,
        min_delta=0.001
    )

    # -----------------------------------------------------
    # OUTPUT PATHS
    # -----------------------------------------------------

    output_dir = (
        ROOT_DIR
        / config["paths"]["model_output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_output_path = (
        output_dir
        / config["paths"]["model_name"]
    )

    history = {
        "train_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_acc = 0.0

    epochs = int(config["training"]["epochs"])

    print("🔄 Training started...")

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    for epoch in range(epochs):

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            device=device
        )

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        # -------------------------------------------------
        # HISTORY
        # -------------------------------------------------

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch+1:2d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # -------------------------------------------------
        # SAVE BEST MODEL
        # -------------------------------------------------

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                model_output_path
            )

            print(
                f"💾 New best model: "
                f"{val_acc:.4f}"
            )

        # -------------------------------------------------
        # EARLY STOPPING
        # -------------------------------------------------

        early_stopping(val_acc, model)

        if early_stopping.early_stop:

            if early_stopping.restore_best:

                model.load_state_dict(
                    early_stopping.best_weights
                )

                print("🔄 Restored best weights")

            break

    # =====================================================
    # FINAL EVALUATION
    # =====================================================

    final_acc, conf_matrix = evaluate_detailed(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=class_names
    )

    print("\n=== FINAL EVALUATION ===")
    print(f"Accuracy: {final_acc:.4f}")

    # =====================================================
    # PLOT TRAINING CURVE
    # =====================================================

    chart_path = (
        output_dir
        / "training_curve.png"
    )

    plot_training_curve(
        history,
        chart_path
    )

    # =====================================================
    # EXPORT TRAINING INFO
    # =====================================================

    end_time = time.time()

    training_info = {

        "model_name":
            config["paths"]["model_name"],

        "classes":
            class_names,

        "best_accuracy":
            float(best_acc),

        "final_accuracy":
            float(final_acc),

        "epochs_trained":
            epoch + 1,

        "device":
            str(device),

        "start_time":
            datetime.fromtimestamp(
                start_time
            ).isoformat(),

        "end_time":
            datetime.fromtimestamp(
                end_time
            ).isoformat(),

        "duration_sec":
            round(end_time - start_time, 2)
    }

    info_path = (
        output_dir
        / "training_info.json"
    )

    info_path.write_text(
        json.dumps(
            training_info,
            indent=2
        )
    )

    print("\n✅ Training completed")
    print(f"📁 Model saved: {model_output_path}")
    print(f"📊 Best Accuracy: {best_acc:.4f}")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    CONFIG_PATH = (
        ROOT_DIR
        / "core"
        / "face_mask_detection"
        / "configs"
        / "config.yml"
    )

    train(CONFIG_PATH)