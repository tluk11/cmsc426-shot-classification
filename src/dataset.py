"""
PyTorch Dataset for basketball shot classification.

Loads pre-extracted frames + labels and serves (image, label) pairs for training.
Used by all training scripts.

Three frame modes:
  - "middle": just the middle frame (default, baseline)
  - "all": all 3 frames stacked on channel dim (3 frames * 3 channels = 9 channels)
  - "random": pick one of the 3 frames at random per epoch (cheap augmentation)

The class indexing is fixed so test predictions stay comparable across runs.
"""

from pathlib import Path
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# Fixed class ordering. Don't change this after training starts.
CLASS_NAMES = ["dunk", "jumpshot", "layup", "three_pointer"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

# The labels.csv uses different formatting; map CSV labels -> our class names.
CSV_LABEL_TO_CLASS = {
    "dunk": "dunk",
    "jumpshot": "jumpshot",
    "layup": "layup",
    "3-pointer": "three_pointer",
    "three_pointer": "three_pointer",
}


def get_train_transforms(image_size=224):
    """Standard augmentations for training images."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(  # ImageNet stats — match pretrained model expectations
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms(image_size=224):
    """No augmentation for val/test."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class BasketballFrameDataset(Dataset):
    """
    Dataset that loads single frames per clip.

    Args:
        split_csv: path to one of train.csv / val.csv / test.csv
        frames_dir: where extracted frames live (e.g. data/frames/)
        transform: torchvision transform to apply
        frame_mode: "middle" | "random" | int (0/1/2 for specific frame index)
    """

    def __init__(self, split_csv, frames_dir, transform=None, frame_mode="middle"):
        self.split_csv = Path(split_csv)
        self.frames_dir = Path(frames_dir)
        self.transform = transform
        self.frame_mode = frame_mode

        df = pd.read_csv(self.split_csv)

        # Map CSV labels to class indices, drop rows with unknown labels
        records = []
        for _, row in df.iterrows():
            csv_label = str(row["shot_type"]).strip()
            class_name = CSV_LABEL_TO_CLASS.get(csv_label)
            if class_name is None:
                continue  # silently drop (e.g., free throw)
            records.append({
                "filename": row["filename"],
                "stem": Path(row["filename"]).stem,
                "label": CLASS_TO_IDX[class_name],
                "class_name": class_name,
            })
        self.records = records

        if not self.records:
            raise RuntimeError(f"No usable rows in {split_csv}")

    def __len__(self):
        return len(self.records)

    def _frame_path(self, stem):
        if self.frame_mode == "middle":
            idx = 1
        elif self.frame_mode == "random":
            idx = random.randint(0, 2)
        elif isinstance(self.frame_mode, int):
            idx = self.frame_mode
        else:
            raise ValueError(f"Unknown frame_mode: {self.frame_mode}")
        return self.frames_dir / stem / f"frame_{idx}.jpg"

    def __getitem__(self, i):
        rec = self.records[i]
        path = self._frame_path(rec["stem"])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, rec["label"]


def get_class_weights(split_csv):
    """
    Inverse-frequency class weights for imbalanced training.
    Pass these to nn.CrossEntropyLoss(weight=...) to give dunks more weight.
    """
    df = pd.read_csv(split_csv)
    counts = torch.zeros(len(CLASS_NAMES))
    for _, row in df.iterrows():
        class_name = CSV_LABEL_TO_CLASS.get(str(row["shot_type"]).strip())
        if class_name is not None:
            counts[CLASS_TO_IDX[class_name]] += 1
    # Inverse frequency, normalized so weights average to 1
    weights = 1.0 / counts.clamp(min=1)
    weights = weights * len(CLASS_NAMES) / weights.sum()
    return weights


if __name__ == "__main__":
    # Quick smoke test
    import sys
    splits_dir = Path("data/splits")
    frames_dir = Path("data/frames")

    if not splits_dir.exists():
        print("Run make_split.py first.")
        sys.exit(1)
    if not frames_dir.exists():
        print("Run extract_frames.py first.")
        sys.exit(1)

    ds = BasketballFrameDataset(
        split_csv=splits_dir / "train.csv",
        frames_dir=frames_dir,
        transform=get_eval_transforms(),
        frame_mode="middle",
    )
    print(f"Train dataset size: {len(ds)}")
    img, label = ds[0]
    print(f"Sample 0: image shape={img.shape}, label={label} ({IDX_TO_CLASS[label]})")
    print(f"Class weights: {get_class_weights(splits_dir / 'train.csv')}")
