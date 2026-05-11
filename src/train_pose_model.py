"""
Train a bidirectional LSTM on pose keypoint sequences.

This is the THIRD of three approaches: pure body mechanics. The model never
sees the image - just the (16, 99) sequence of normalized keypoints. If this
beats random, body posture alone carries enough information to discriminate
shot types.

Note: trains from scratch (no pretrained weights) - this is small enough that
we don't need them. Should train fast (~1 sec/epoch on GPU).

Usage (from final_project/):
    python src/train_pose_model.py
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
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from pose_dataset import (
    BasketballPoseDataset,
    CLASS_NAMES,
    get_class_weights,
)


DEFAULTS = {
    "splits_dir": "data/splits",
    "pose_dir": "data/pose",
    "results_dir": "results/runs",
    "run_name": "pose_baseline",
    "epochs": 60,            # train from scratch - more epochs
    "batch_size": 16,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.4,
    "num_workers": 0,
    "seed": 42,
}


class PoseLSTM(nn.Module):
    """Bi-LSTM over the 16-step pose sequence, mean-pool, classify."""

    def __init__(self, input_size=99, hidden_size=128, num_layers=2,
                 num_classes=4, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: (B, T, 99)
        out, _ = self.lstm(x)         # (B, T, 2*hidden)
        pooled = out.mean(dim=1)      # mean over time
        pooled = self.norm(pooled)
        return self.head(pooled)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device); labels = labels.to(device)
            logits = model(x)
            loss = criterion(logits, labels)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return (total_loss / max(n, 1),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average="macro", zero_division=0))


def main():
    parser = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    splits_dir = Path(args.splits_dir)
    pose_dir = Path(args.pose_dir)
    run_dir = Path(args.results_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not pose_dir.exists():
        print(f"ERROR: {pose_dir} not found. Run extract_pose.py first.")
        sys.exit(1)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = BasketballPoseDataset(splits_dir / "train.csv", pose_dir, augment=True)
    val_ds = BasketballPoseDataset(splits_dir / "val.csv", pose_dir, augment=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = PoseLSTM(
        input_size=99,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=len(CLASS_NAMES),
        dropout=args.dropout,
    ).to(device)

    class_weights = get_class_weights(splits_dir / "train.csv").to(device)
    print(f"Class weights ({CLASS_NAMES}): {class_weights.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    metrics_rows = []
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        n = 0

        for x, labels in train_loader:
            x = x.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds = logits.argmax(dim=1).detach()
            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

        scheduler.step()
        train_loss /= max(n, 1)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)
        writer.add_scalar("f1_macro/val", val_f1, epoch)

        metrics_rows.append({
            "epoch": epoch,
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

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc, "val_f1": val_f1,
        "config": vars(args),
    }, run_dir / "last_model.pth")

    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    writer.close()

    print(f"\nDone. Best val_acc = {best_val_acc:.3f} at epoch {best_epoch}.")
    print(f"Run artifacts: {run_dir.resolve()}")
    print(f"\nNext: python src/eval_pose_model.py --run_name {args.run_name}")


if __name__ == "__main__":
    main()
