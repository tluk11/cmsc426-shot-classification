"""
PyTorch Dataset for video model: returns (T, C, H, W) tensors.

Loads 16 pre-extracted frames per clip and stacks them into a video tensor
formatted for torchvision's R(2+1)D-18 (or similar) Kinetics-pretrained models.

Frame normalization uses the Kinetics mean/std the pretrained model expects.
"""

from pathlib import Path
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# Class config copied from frame dataset for consistency
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

# Kinetics-400 normalization stats (used by R(2+1)D-18 pretrained weights)
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

N_FRAMES = 16  # number of frames per clip (must match extract_video_frames.py)


def get_train_video_transforms(size=112):
    """Per-frame transforms applied uniformly across the 16 frames."""
    return transforms.Compose([
        transforms.Resize((size + 16, size + 16)),
        transforms.CenterCrop(size),  # use center crop for stability with small data
        transforms.ToTensor(),
        transforms.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
    ])


def get_eval_video_transforms(size=112):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
    ])


class BasketballVideoDataset(Dataset):
    """
    Returns (video_tensor, label) where video_tensor has shape (C, T, H, W)
    matching torchvision video models' expected input.

    Args:
        split_csv: train/val/test CSV
        video_frames_dir: where 16 frames per clip live
        transform: per-frame torchvision transform
        n_frames: how many frames to use (default 16)
        random_flip: horizontally flip the whole clip with 50% probability (train only)
    """

    def __init__(self, split_csv, video_frames_dir, transform=None,
                 n_frames=N_FRAMES, random_flip=False):
        self.split_csv = Path(split_csv)
        self.video_frames_dir = Path(video_frames_dir)
        self.transform = transform
        self.n_frames = n_frames
        self.random_flip = random_flip

        df = pd.read_csv(self.split_csv)
        records = []
        for _, row in df.iterrows():
            csv_label = str(row["shot_type"]).strip()
            class_name = CSV_LABEL_TO_CLASS.get(csv_label)
            if class_name is None:
                continue
            records.append({
                "filename": row["filename"],
                "stem": Path(row["filename"]).stem,
                "label": CLASS_TO_IDX[class_name],
            })
        self.records = records
        if not records:
            raise RuntimeError(f"No usable rows in {split_csv}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        clip_dir = self.video_frames_dir / rec["stem"]

        # Decide augmentation for this sample (applied consistently across frames)
        do_flip = self.random_flip and random.random() < 0.5

        frames = []
        for j in range(self.n_frames):
            path = clip_dir / f"frame_{j:02d}.jpg"
            img = Image.open(path).convert("RGB")
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Stack into (T, C, H, W), then permute to (C, T, H, W)
        video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        return video, rec["label"]


def get_class_weights(split_csv):
    """Same inverse-frequency weighting as frame dataset."""
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
    # Smoke test
    import sys
    splits_dir = Path("data/splits")
    video_dir = Path("data/video_frames")
    if not splits_dir.exists() or not video_dir.exists():
        print("Run make_split.py and extract_video_frames.py first.")
        sys.exit(1)
    ds = BasketballVideoDataset(
        split_csv=splits_dir / "train.csv",
        video_frames_dir=video_dir,
        transform=get_eval_video_transforms(),
    )
    print(f"Train video dataset size: {len(ds)}")
    video, label = ds[0]
    print(f"Sample 0: video shape={tuple(video.shape)} (expect C=3, T=16, H=112, W=112)")
    print(f"Label: {label} ({IDX_TO_CLASS[label]})")
