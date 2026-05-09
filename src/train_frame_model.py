"""
Train a ResNet-50 frame-based classifier on basketball shot clips.

This is the BASELINE for the project: takes a single frame per clip and predicts
the shot type. Uses ImageNet-pretrained weights and fine-tunes on our data.

Usage (from final_project/):
    python src/train_frame_model.py
    python src/train_frame_model.py --epochs 30 --lr 1e-4 --run_name frame_baseline_v2

Outputs:
    results/runs/{run_name}/
        best_model.pth       - checkpoint with best val accuracy
        last_model.pth       - checkpoint at end of training
        metrics.csv          - per-epoch train/val loss + accuracy
        config.json          - hyperparameters used for this run
        tensorboard/         - TensorBoard event files

To watch training live, in a SECOND terminal (also venv-activated):
    tensorboard --logdir results/runs
Then open http://localhost:6006 in your browser.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    BasketballFrameDataset,
    CLASS_NAMES,
    get_class_weights,
    get_train_transforms,
    get_eval_transforms,
)


# ----- Defaults -----
DEFAULTS = {
    "splits_dir": "data/splits",
    "frames_dir": "data/frames",
    "results_dir": "results/runs",
    "run_name": "frame_baseline",
    "epochs": 25,
    "batch_size": 16,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "image_size": 224,
    "frame_mode": "middle",
    "num_workers": 0,    # Windows + Python on small datasets: 0 is safer
    "freeze_backbone_epochs": 3,  # train head only for first N epochs
    "seed": 42,
}


def build_model(num_classes):
    """ResNet-50 with ImageNet weights, classifier replaced for our N classes."""
    weights = ResNet50_Weights.IMAGENET1K_V2  # the better of the two ImageNet weights
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable):
    """Freeze/unfreeze everything except the final fc layer."""
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def evaluate(model, loader, criterion, device):
    """Run one pass over a loader, return (loss, accuracy, macro_f1, all_preds, all_labels)."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(n_samples, 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        parser.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # Paths
    splits_dir = Path(args.splits_dir)
    frames_dir = Path(args.frames_dir)
    run_dir = Path(args.results_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tensorboard"

    if not (splits_dir / "train.csv").exists():
        print(f"ERROR: {splits_dir/'train.csv'} not found. Run make_split.py first.")
        sys.exit(1)
    if not frames_dir.exists():
        print(f"ERROR: {frames_dir} not found. Run extract_frames.py first.")
        sys.exit(1)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ----- Data -----
    train_ds = BasketballFrameDataset(
        split_csv=splits_dir / "train.csv",
        frames_dir=frames_dir,
        transform=get_train_transforms(args.image_size),
        frame_mode=args.frame_mode,
    )
    val_ds = BasketballFrameDataset(
        split_csv=splits_dir / "val.csv",
        frames_dir=frames_dir,
        transform=get_eval_transforms(args.image_size),
        frame_mode="middle",
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ----- Model -----
    model = build_model(num_classes=len(CLASS_NAMES)).to(device)

    # Class weights for imbalanced dunks
    class_weights = get_class_weights(splits_dir / "train.csv").to(device)
    print(f"Class weights ({CLASS_NAMES}): {class_weights.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ----- TensorBoard -----
    writer = SummaryWriter(log_dir=str(tb_dir))

    # ----- Training loop -----
    metrics_rows = []
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        # Phase 1 (early epochs): train only the new head with backbone frozen
        # Phase 2 (later epochs): unfreeze whole network
        if epoch <= args.freeze_backbone_epochs:
            set_backbone_trainable(model, trainable=False)
            phase = "head-only"
        elif epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            phase = "full"
        else:
            phase = "full"

        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [{phase}]")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            preds = logits.argmax(dim=1).detach()
            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()
        train_loss /= max(n, 1)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)

        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        # Log
        print(f"  train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("f1_macro/val", val_f1, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        metrics_rows.append({
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "config": vars(args),
            }, run_dir / "best_model.pth")
            print(f"  -> new best val_acc, saved.")

    # Final save
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "val_f1": val_f1,
        "config": vars(args),
    }, run_dir / "last_model.pth")

    # Save metrics CSV
    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    writer.close()

    print(f"\nDone. Best val_acc = {best_val_acc:.3f} at epoch {best_epoch}.")
    print(f"Run artifacts: {run_dir.resolve()}")
    print(f"\nNext: evaluate on test set with src/eval_frame_model.py")


if __name__ == "__main__":
    main()
