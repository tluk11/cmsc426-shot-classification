"""
Generate a stratified train/val/test split (60/20/20) and save to CSVs.

Stratified means each split has roughly the same class distribution as the
full dataset. With class imbalance (dunks at 25 vs others at 60), this
matters - a random split could easily put 0 dunks in test.

Run once. The split is saved to data/splits/ and committed to git so both
team members use the exact same split. Re-running with the same seed is
deterministic, but don't re-run after you've started training - results
across runs won't be comparable.

Usage:
    python src/make_split.py
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Default paths assume we run from final_project/ directory
DEFAULT_LABELS = Path("../Final Project NBA dataset/labeled/labels.csv")
DEFAULT_OUT_DIR = Path("data/splits")
RANDOM_SEED = 42  # never change this


def main():
    parser = argparse.ArgumentParser(description="Make stratified train/val/test split.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS,
                        help="Path to labels.csv")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Where to write split CSVs")
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--drop_classes", nargs="*", default=["free throw"],
                        help="Classes to exclude from the dataset")
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"ERROR: labels file not found at {args.labels.resolve()}")
        print("Run this script from the final_project/ directory, "
              "or pass --labels with the correct path.")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(args.labels)
    print(f"Loaded {len(df)} rows from {args.labels}")

    # Drop excluded classes (e.g., free throw)
    if args.drop_classes:
        before = len(df)
        df = df[~df["shot_type"].isin(args.drop_classes)].reset_index(drop=True)
        print(f"Dropped {before - len(df)} rows with classes {args.drop_classes}")

    # Class distribution check
    print("\nClass distribution:")
    print(df["shot_type"].value_counts().to_string())

    # Stratified split: first carve out test, then split remainder into train/val.
    # We adjust val_size to be relative to the remaining train+val portion.
    test_frac = args.test_size
    val_frac_of_remainder = args.val_size / (1.0 - test_frac)

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        stratify=df["shot_type"],
        random_state=RANDOM_SEED,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_frac_of_remainder,
        stratify=train_val_df["shot_type"],
        random_state=RANDOM_SEED,
    )

    # Sort each by filename for stable ordering
    train_df = train_df.sort_values("filename").reset_index(drop=True)
    val_df = val_df.sort_values("filename").reset_index(drop=True)
    test_df = test_df.sort_values("filename").reset_index(drop=True)

    # Save
    train_path = args.out_dir / "train.csv"
    val_path = args.out_dir / "val.csv"
    test_path = args.out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nWrote splits to {args.out_dir}/")
    print(f"  train.csv: {len(train_df)} clips")
    print(f"  val.csv:   {len(val_df)} clips")
    print(f"  test.csv:  {len(test_df)} clips")

    # Per-class breakdown of each split
    print("\nPer-split class counts:")
    summary = pd.DataFrame({
        "train": train_df["shot_type"].value_counts(),
        "val": val_df["shot_type"].value_counts(),
        "test": test_df["shot_type"].value_counts(),
    }).fillna(0).astype(int)
    summary["total"] = summary.sum(axis=1)
    print(summary.to_string())

    # Sanity: no overlap between splits
    train_files = set(train_df["filename"])
    val_files = set(val_df["filename"])
    test_files = set(test_df["filename"])
    assert not (train_files & val_files), "Train/val overlap!"
    assert not (train_files & test_files), "Train/test overlap!"
    assert not (val_files & test_files), "Val/test overlap!"
    print("\nNo overlap between splits. Good.")


if __name__ == "__main__":
    main()
