"""
Evaluate a trained video model on the test set.

Same outputs as eval_frame_model.py: confusion matrix, classification report,
per-clip predictions, summary JSON.

Usage (from final_project/):
    python src/eval_video_model.py
    python src/eval_video_model.py --run_name video_baseline_v2
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from video_dataset import (
    BasketballVideoDataset,
    CLASS_NAMES, IDX_TO_CLASS,
    get_eval_video_transforms,
)
from train_video_model import build_model


def plot_confusion_matrix(cm, class_names, out_path, title="Confusion matrix"):
    cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            t = f"{cm[i, j]}\n({cm_norm[i, j]:.0%})"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, t, ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="video_baseline")
    parser.add_argument("--results_dir", default="results/runs")
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--video_frames_dir", default="data/video_frames")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint", default="best_model.pth")
    args = parser.parse_args()

    run_dir = Path(args.results_dir) / args.run_name
    ckpt_path = run_dir / args.checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    image_size = config.get("image_size", 112)

    model = build_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.3f})")

    splits_dir = Path(args.splits_dir)
    video_frames_dir = Path(args.video_frames_dir)
    ds = BasketballVideoDataset(
        split_csv=splits_dir / f"{args.split}.csv",
        video_frames_dir=video_frames_dir,
        transform=get_eval_video_transforms(image_size),
        random_flip=False,
    )
    loader = DataLoader(ds, batch_size=config.get("batch_size", 4), shuffle=False)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            logits = model(videos)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n=== {args.split.upper()} results for run '{args.run_name}' ===")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds,
                                labels=list(range(len(CLASS_NAMES))),
                                target_names=CLASS_NAMES,
                                zero_division=0, digits=3))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    cm_path = run_dir / f"confusion_matrix_{args.split}.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path,
                          title=f"{args.run_name} - {args.split} (acc={acc:.3f})")
    print(f"\nConfusion matrix saved: {cm_path}")

    rows = []
    for i, rec in enumerate(ds.records):
        rows.append({
            "filename": rec["filename"],
            "true_label": IDX_TO_CLASS[rec["label"]],
            "pred_label": IDX_TO_CLASS[all_preds[i]],
            "correct": all_preds[i] == rec["label"],
            "confidence": float(all_probs[i].max()),
            **{f"prob_{c}": float(all_probs[i, idx]) for idx, c in enumerate(CLASS_NAMES)},
        })
    pred_df = pd.DataFrame(rows)
    pred_path = run_dir / f"predictions_{args.split}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved: {pred_path}")

    summary = {
        "run_name": args.run_name, "split": args.split, "checkpoint": args.checkpoint,
        "epoch": int(ckpt["epoch"]),
        "accuracy": float(acc), "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
        "n_samples": len(all_labels),
        "confusion_matrix": cm.tolist(), "class_names": CLASS_NAMES,
    }
    with open(run_dir / f"summary_{args.split}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {run_dir / f'summary_{args.split}.json'}")

    wrong = pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False)
    if len(wrong) > 0:
        print(f"\nTop 10 most-confident wrong predictions:")
        print(wrong[["filename", "true_label", "pred_label", "confidence"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
