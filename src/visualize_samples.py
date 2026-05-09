"""
Sanity-check the dataset pipeline. Loads a random batch and saves a labeled grid.

This is the step where you catch:
  - Frames extracted from the wrong moment (no shot visible)
  - Letterboxing / black bars hurting crop quality
  - Mislabeled clips
  - Aspect ratio weirdness from resizing

Usage (from final_project/):
    python src/visualize_samples.py
    python src/visualize_samples.py --split val --n 16
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    BasketballFrameDataset,
    IDX_TO_CLASS,
    get_eval_transforms,
)


def denormalize(tensor):
    """Reverse the ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--n", type=int, default=16, help="Number of samples to show")
    parser.add_argument("--out", type=Path, default=Path("results/sample_grid.png"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    splits_dir = Path("data/splits")
    frames_dir = Path("data/frames")

    if not (splits_dir / f"{args.split}.csv").exists():
        print(f"ERROR: {splits_dir / f'{args.split}.csv'} not found. Run make_split.py first.")
        sys.exit(1)

    ds = BasketballFrameDataset(
        split_csv=splits_dir / f"{args.split}.csv",
        frames_dir=frames_dir,
        transform=get_eval_transforms(image_size=224),
        frame_mode="middle",
    )
    print(f"Loaded {args.split} set: {len(ds)} clips")

    # Pick n random samples
    indices = random.sample(range(len(ds)), min(args.n, len(ds)))
    samples = [ds[i] for i in indices]
    files = [ds.records[i]["filename"] for i in indices]

    # Build grid
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    for ax, (img, label), fname in zip(axes, samples, files):
        img_disp = denormalize(img).permute(1, 2, 0).numpy()
        ax.imshow(img_disp)
        ax.set_title(
            f"{IDX_TO_CLASS[label]}\n{fname}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved grid to: {args.out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
