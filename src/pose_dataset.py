"""
PyTorch Dataset for pose-based shot classification.

Loads pre-extracted MediaPipe keypoints + a normalization step that makes the
representation invariant to where the person stands in the frame and their size.

Normalization:
    - Center: subtract the hip-center (landmark 23/24 midpoint) per frame
    - Scale: divide by torso length (hip-center to shoulder-center) per frame
This way the model focuses on body mechanics, not court position or zoom level.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ["dunk", "jumpshot", "layup", "three_pointer"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}
CSV_LABEL_TO_CLASS = {
    "dunk": "dunk",
    "jumpshot": "jumpshot",
    "layup": "layup",
    "3-pointer": "three_pointer",
    "three_pointer": "three_pointer",
}

# MediaPipe Pose landmark indices we care about
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

N_FRAMES = 16
N_LANDMARKS = 33
N_COORDS = 3  # we'll keep x, y, z (drop visibility from features but keep mask)


def normalize_pose_sequence(keypoints):
    """
    Normalize a (T, 33, 4) keypoint sequence: hip-center origin, torso-length scale.

    Returns:
        normalized: (T, 33, 3) array (x, y, z)
        valid_mask: (T,) bool - True if normalization was reliable for that frame
    """
    T = keypoints.shape[0]
    coords = keypoints[..., :3]  # (T, 33, 3)
    visibility = keypoints[..., 3]  # (T, 33)

    out = np.zeros((T, N_LANDMARKS, N_COORDS), dtype=np.float32)
    valid_mask = np.zeros(T, dtype=bool)

    for t in range(T):
        frame = coords[t]
        vis = visibility[t]

        # If hips and shoulders aren't visible enough, skip normalization
        critical_pts = [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER]
        if any(vis[p] < 0.3 for p in critical_pts):
            continue

        hip_center = (frame[LEFT_HIP] + frame[RIGHT_HIP]) / 2
        shoulder_center = (frame[LEFT_SHOULDER] + frame[RIGHT_SHOULDER]) / 2
        torso_len = np.linalg.norm(shoulder_center[:2] - hip_center[:2])
        if torso_len < 1e-4:
            continue

        centered = frame - hip_center
        out[t] = centered / torso_len
        valid_mask[t] = True

    # For frames that failed normalization, fill with nearest valid frame
    if valid_mask.any():
        valid_indices = np.where(valid_mask)[0]
        for t in range(T):
            if not valid_mask[t]:
                # Use closest valid frame
                nearest = valid_indices[np.argmin(np.abs(valid_indices - t))]
                out[t] = out[nearest]

    return out, valid_mask


class BasketballPoseDataset(Dataset):
    """
    Returns (pose_tensor, label) where pose_tensor has shape (T, 33*3) = (16, 99).
    Flattened so we can feed it directly into an LSTM as feature vectors per timestep.
    """

    def __init__(self, split_csv, pose_dir, augment=False):
        self.split_csv = Path(split_csv)
        self.pose_dir = Path(pose_dir)
        self.augment = augment

        df = pd.read_csv(self.split_csv)
        records = []
        for _, row in df.iterrows():
            csv_label = str(row["shot_type"]).strip()
            class_name = CSV_LABEL_TO_CLASS.get(csv_label)
            if class_name is None:
                continue
            stem = Path(row["filename"]).stem
            pose_path = self.pose_dir / f"{stem}.npz"
            if not pose_path.exists():
                continue
            records.append({
                "filename": row["filename"],
                "stem": stem,
                "label": CLASS_TO_IDX[class_name],
                "pose_path": pose_path,
            })
        self.records = records
        if not records:
            raise RuntimeError(f"No usable pose data for {split_csv}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        data = np.load(rec["pose_path"])
        keypoints = data["keypoints"]  # (16, 33, 4)

        normalized, _ = normalize_pose_sequence(keypoints)  # (16, 33, 3)

        # Augmentation: horizontal flip (negate x) - basketball is largely symmetric
        if self.augment and np.random.rand() < 0.5:
            normalized[..., 0] = -normalized[..., 0]

        # Augmentation: small noise injection
        if self.augment:
            normalized = normalized + np.random.randn(*normalized.shape).astype(np.float32) * 0.01

        # Flatten to (T, 99)
        flat = normalized.reshape(N_FRAMES, -1)
        return torch.from_numpy(flat).float(), rec["label"]


def get_class_weights(split_csv):
    df = pd.read_csv(split_csv)
    counts = torch.zeros(len(CLASS_NAMES))
    for _, row in df.iterrows():
        class_name = CSV_LABEL_TO_CLASS.get(str(row["shot_type"]).strip())
        if class_name is not None:
            counts[CLASS_TO_IDX[class_name]] += 1
    weights = 1.0 / counts.clamp(min=1)
    weights = weights * len(CLASS_NAMES) / weights.sum()
    return weights


if __name__ == "__main__":
    import sys
    splits_dir = Path("data/splits")
    pose_dir = Path("data/pose")
    if not splits_dir.exists() or not pose_dir.exists():
        print("Run make_split.py and extract_pose.py first.")
        sys.exit(1)
    ds = BasketballPoseDataset(splits_dir / "train.csv", pose_dir, augment=False)
    print(f"Train pose dataset size: {len(ds)}")
    x, label = ds[0]
    print(f"Sample 0: pose shape={tuple(x.shape)} (expect T=16, F=99)")
    print(f"Label: {label} ({IDX_TO_CLASS[label]})")
