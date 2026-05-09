"""
Diagnostic: show all 3 extracted frames side-by-side per clip.

Use this to decide:
  - Is at least one of the 3 frames capturing the shot? (extraction OK, just need to pick the right index)
  - Are all 3 frames missing the shot? (re-trimming needed)
  - Is the clip just labeled wrong? (re-label)

Usage (from final_project/):
    python src/visualize_triplets.py
    python src/visualize_triplets.py --split val --n 8
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


CSV_LABEL_TO_CLASS = {
    "dunk": "dunk",
    "jumpshot": "jumpshot",
    "layup": "layup",
    "3-pointer": "three_pointer",
    "three_pointer": "three_pointer",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--n", type=int, default=8, help="Number of clips to show")
    parser.add_argument("--out", type=Path, default=Path("results/triplet_grid.png"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class_filter", default=None,
                        help="Show only this class (e.g., 'jumpshot')")
    args = parser.parse_args()

    random.seed(args.seed)

    splits_dir = Path("data/splits")
    frames_dir = Path("data/frames")

    csv_path = splits_dir / f"{args.split}.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if args.class_filter:
        df = df[df["shot_type"].apply(
            lambda s: CSV_LABEL_TO_CLASS.get(str(s).strip()) == args.class_filter
        )]
        if len(df) == 0:
            print(f"No clips with class '{args.class_filter}' in {args.split}.")
            sys.exit(1)

    n = min(args.n, len(df))
    sampled = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    # 3 columns (frames), n rows (clips)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row_i, row in sampled.iterrows():
        filename = row["filename"]
        stem = Path(filename).stem
        label = row["shot_type"]

        for col_i in range(3):
            ax = axes[row_i, col_i]
            frame_path = frames_dir / stem / f"frame_{col_i}.jpg"
            if frame_path.exists():
                img = Image.open(frame_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center")
            ax.axis("off")
            if col_i == 0:
                ax.set_ylabel(f"{label}\n{filename}", fontsize=9, rotation=0,
                              labelpad=80, va="center", ha="right")
            ax.set_title(f"frame_{col_i} ({['25%', '50%', '75%'][col_i]})", fontsize=9)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"Saved triplet grid to: {args.out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
