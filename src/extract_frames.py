"""
Extract 3 evenly-spaced frames from each clip in the dataset.

For a clip with N frames, extracts frames at positions 0.25, 0.5, 0.75 (start,
middle, end of the action). Saved as JPEG under data/frames/<filename>/frame_0.jpg,
frame_1.jpg, frame_2.jpg.

The frame-based baseline can train on just frame_1 (middle). Variants can use
all three.

Run once. Idempotent - skips clips already processed.

Usage (from final_project/ directory):
    python src/extract_frames.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


DEFAULT_CLIPS_DIR = Path("../Final Project NBA dataset/labeled")
DEFAULT_OUT_DIR = Path("data/frames")
FRAME_FRACTIONS = [0.25, 0.5, 0.75]  # which frames to grab (as fraction of clip)


def extract_clip_frames(clip_path, out_dir, fractions=FRAME_FRACTIONS, force=False):
    """
    Extract frames from one clip. Returns (success, n_extracted, error_msg).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if all frames already exist
    target_paths = [out_dir / f"frame_{i}.jpg" for i in range(len(fractions))]
    if not force and all(p.exists() for p in target_paths):
        return True, 0, None

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return False, 0, "Could not open video"

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames < 3:
        cap.release()
        return False, 0, f"Only {n_frames} frames in clip"

    extracted = 0
    for i, frac in enumerate(fractions):
        target_idx = max(0, min(n_frames - 1, int(n_frames * frac)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return False, extracted, f"Failed to read frame {target_idx}"
        out_path = out_dir / f"frame_{i}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        extracted += 1

    cap.release()
    return True, extracted, None


def main():
    parser = argparse.ArgumentParser(description="Extract frames from clips.")
    parser.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--labels", type=Path, default=None,
                        help="Optional labels.csv path (default: clips_dir/labels.csv)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if frames already exist")
    args = parser.parse_args()

    if not args.clips_dir.exists():
        print(f"ERROR: clips dir not found at {args.clips_dir.resolve()}")
        sys.exit(1)

    labels_path = args.labels or (args.clips_dir / "labels.csv")
    if not labels_path.exists():
        print(f"ERROR: labels.csv not found at {labels_path.resolve()}")
        sys.exit(1)

    df = pd.read_csv(labels_path)
    print(f"Found {len(df)} clips in labels.csv")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_skipped = 0
    failures = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        filename = row["filename"]
        clip_path = args.clips_dir / filename
        if not clip_path.exists():
            failures.append((filename, "file missing"))
            continue

        clip_out_dir = args.out_dir / Path(filename).stem
        success, n_extracted, err = extract_clip_frames(
            clip_path, clip_out_dir, force=args.force
        )
        if not success:
            failures.append((filename, err))
        elif n_extracted == 0:
            n_skipped += 1
        else:
            n_success += 1

    print(f"\nExtracted frames for {n_success} new clips")
    print(f"Skipped (already done) {n_skipped} clips")
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for fname, err in failures[:20]:
            print(f"  {fname}: {err}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    print(f"\nFrames written to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
