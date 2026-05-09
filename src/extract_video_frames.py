"""
Pre-extract 16 evenly-spaced frames per clip for video model training.

Saved as JPEG sequences under data/video_frames/<clip_stem>/frame_NN.jpg.
This is way faster than decoding video on every training iteration.

Run once. Idempotent - skips clips already processed.

Usage (from final_project/):
    python src/extract_video_frames.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_CLIPS_DIR = Path("../Final Project NBA dataset/labeled")
DEFAULT_OUT_DIR = Path("data/video_frames")
N_FRAMES = 16  # what R(2+1)D-18 expects


def extract_video_frames(clip_path, out_dir, n_frames=N_FRAMES, force=False):
    """Extract n_frames evenly-spaced frames. Returns (success, error_msg)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    target_paths = [out_dir / f"frame_{i:02d}.jpg" for i in range(n_frames)]
    if not force and all(p.exists() for p in target_paths):
        return True, None

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return False, "Could not open video"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        # Repeat last frame if too short
        target_indices = np.linspace(0, max(0, total - 1), n_frames).astype(int)
    else:
        target_indices = np.linspace(0, total - 1, n_frames).astype(int)

    for i, target_idx in enumerate(target_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return False, f"Failed to read frame {target_idx}"
        cv2.imwrite(str(out_dir / f"frame_{i:02d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    cap.release()
    return True, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.clips_dir.exists():
        print(f"ERROR: clips dir not found at {args.clips_dir.resolve()}")
        sys.exit(1)

    labels_path = args.labels or (args.clips_dir / "labels.csv")
    df = pd.read_csv(labels_path)
    print(f"Found {len(df)} clips")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    failures = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        filename = row["filename"]
        clip_path = args.clips_dir / filename
        if not clip_path.exists():
            failures.append((filename, "missing"))
            continue
        clip_out = args.out_dir / Path(filename).stem
        ok, err = extract_video_frames(clip_path, clip_out, force=args.force)
        if not ok:
            failures.append((filename, err))
        else:
            n_done += 1

    print(f"\nProcessed {n_done} clips")
    if failures:
        print(f"Failures: {len(failures)}")
        for fname, err in failures[:10]:
            print(f"  {fname}: {err}")
    print(f"Frames written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
