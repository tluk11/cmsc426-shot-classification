"""
Extract MediaPipe Pose keypoints from each clip.

For each clip, runs MediaPipe Pose on 16 evenly-spaced frames and saves
a (16, 33, 4) array - 16 frames, 33 keypoints, [x, y, z, visibility].
Also saves a (16,) visibility mask indicating which frames had detections.

Saved as .npz under data/pose/<clip_stem>.npz.

If a frame has no person detected, the keypoints are filled from the last
detection (or zeros if no detection yet). Frame visibility is logged so the
training loop can ignore unreliable frames.

Usage (from final_project/):
    python src/extract_pose.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_CLIPS_DIR = Path("../Final Project NBA dataset/labeled")
DEFAULT_OUT_DIR = Path("data/pose")
N_FRAMES = 16
N_LANDMARKS = 33  # MediaPipe Pose has 33 body landmarks


def extract_pose_for_clip(clip_path, n_frames=N_FRAMES):
    """
    Returns:
        keypoints: (n_frames, 33, 4) numpy array of [x_norm, y_norm, z, visibility]
                   x and y are normalized to [0, 1] by MediaPipe; z is depth.
        detected_mask: (n_frames,) boolean - whether MediaPipe found a person
        error: None on success, else string
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None, None, "Could not open video"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None, None, "No frames"

    target_indices = np.linspace(0, max(0, total - 1), n_frames).astype(int)

    keypoints = np.zeros((n_frames, N_LANDMARKS, 4), dtype=np.float32)
    detected_mask = np.zeros(n_frames, dtype=bool)
    last_valid = None  # (33, 4) from the most recent successful detection

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,           # 1 is the standard "full" model
        enable_segmentation=False,
        min_detection_confidence=0.3, # lower threshold for crowded basketball scenes
        min_tracking_confidence=0.3,
    )

    try:
        for i, idx in enumerate(target_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                if last_valid is not None:
                    keypoints[i] = last_valid
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                arr = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in result.pose_landmarks.landmark
                ], dtype=np.float32)
                keypoints[i] = arr
                last_valid = arr
                detected_mask[i] = True
            else:
                if last_valid is not None:
                    keypoints[i] = last_valid
    finally:
        pose.close()
        cap.release()

    return keypoints, detected_mask, None


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
    n_skipped = 0
    failures = []
    low_detection = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting pose"):
        filename = row["filename"]
        clip_path = args.clips_dir / filename
        if not clip_path.exists():
            failures.append((filename, "missing"))
            continue
        out_path = args.out_dir / f"{Path(filename).stem}.npz"
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue

        keypoints, detected_mask, err = extract_pose_for_clip(clip_path)
        if err:
            failures.append((filename, err))
            continue

        n_detected = int(detected_mask.sum())
        if n_detected < 4:
            # Less than 4 of 16 frames had a person detected
            low_detection.append((filename, n_detected))

        np.savez_compressed(out_path,
                            keypoints=keypoints,
                            detected_mask=detected_mask)
        n_done += 1

    print(f"\nProcessed {n_done} clips")
    print(f"Skipped (already done) {n_skipped} clips")
    if failures:
        print(f"\nFailures: {len(failures)}")
        for f, err in failures[:10]:
            print(f"  {f}: {err}")
    if low_detection:
        print(f"\nLow-detection clips (<4 frames with pose): {len(low_detection)}")
        for f, n in low_detection[:10]:
            print(f"  {f}: only {n}/16 frames detected")
    print(f"\nPose data saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
