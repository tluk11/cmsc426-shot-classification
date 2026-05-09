"""
Train an R(2+1)D-18 video classifier on basketball shot clips.

This is the SECOND of three approaches: temporal motion. Takes 16 frames per
clip, processes them through a pretrained Kinetics-400 video network, and
predicts the shot type.

The hypothesis: if motion matters (release, follow-through, ball trajectory),
this should beat the frame-only baseline.

Usage (from final_project/):
    python src/train_video_model.py
    python src/train_video_model.py --epochs 25 --batch_size 4 --run_name video_v2
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from video_dataset import (
    BasketballVideoDataset,
    CLASS_NAMES,
    get_class_weights,
    get_train_video_transforms,
    get_eval_video_transforms,
)


DEFAULTS = {
    "splits_dir": "data/splits",
    "video_frames_dir": "data/video_frames",
    "results_dir": "results/runs",
    "run_name": "video_baseline",
    "epochs": 20,
    "batch_size": 4,        # video tensors are big - 4 fits comfortably in 8GB VRAM
    "lr": 1e-4,             # lower LR for video; pretrained Kinetics features are valuable
    "weight_decay": 1e-4,
    "image_size": 112,      # what R(2+1)D-18 expects
    "num_workers": 0,
    "freeze_backbone_epochs": 3,
    "seed": 42,
}


def build_model(num_classes):
    """R(2+1)D-18 with Kinetics-400 weights, classifier replaced."""
    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable):
    """Freeze/unfreeze everything except final fc."""
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(videos)
            loss = criterion(logits, labels)
            total_loss += loss.item() * videos.size(0)
            n += videos.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    splits_dir = Path(args.splits_dir)
    video_frames_dir = Path(args.video_frames_dir)
    run_dir = Path(args.results_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tensorboard"

    if not (splits_dir / "train.csv").exists():
        print(f"ERROR: {splits_dir/'train.csv'} not found.")
        sys.exit(1)
    if not video_frames_dir.exists():
        print(f"ERROR: {video_frames_dir} not found. Run extract_video_frames.py first.")
        sys.exit(1)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = BasketballVideoDataset(
        split_csv=splits_dir / "train.csv",
        video_frames_dir=video_frames_dir,
        transform=get_train_video_transforms(args.image_size),
        random_flip=True,
    )
    val_ds = BasketballVideoDataset(
        split_csv=splits_dir / "val.csv",
        video_frames_dir=video_frames_dir,
        transform=get_eval_video_transforms(args.image_size),
        random_flip=False,
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

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)

    class_weights = get_class_weights(splits_dir / "train.csv").to(device)
    print(f"Class weights ({CLASS_NAMES}): {class_weights.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(log_dir=str(tb_dir))

    metrics_rows = []
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
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
        for videos, labels in pbar:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(videos)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * videos.size(0)
            n += videos.size(0)
            preds = logits.argmax(dim=1).detach()
            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()
        train_loss /= max(n, 1)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(f"  train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("f1_macro/val", val_f1, epoch)

        metrics_rows.append({
            "epoch": epoch, "phase": phase,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        })

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

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "val_f1": val_f1,
        "config": vars(args),
    }, run_dir / "last_model.pth")

    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    writer.close()

    print(f"\nDone. Best val_acc = {best_val_acc:.3f} at epoch {best_epoch}.")
    print(f"Run artifacts: {run_dir.resolve()}")
    print(f"\nNext: python src/eval_video_model.py --run_name {args.run_name}")


if __name__ == "__main__":
    main()
