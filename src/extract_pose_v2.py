"""
Improved pose extraction: detect people first with YOLOv8, pick the most
likely shooter (largest person in lower-middle of frame), crop, then run
MediaPipe on the crop.

This typically gives much cleaner pose detections on broadcast footage
where multiple people are visible.

Saves to data/pose_v2/ so the original pose extraction is preserved.

Usage (from final_project/):
    python src/extract_pose_v2.py
"""

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_CLIPS_DIR = Path("../Final Project NBA dataset/labeled")
DEFAULT_OUT_DIR = Path("data/pose_v2")
N_FRAMES = 16
N_LANDMARKS = 33


def pick_best_person(boxes, frame_shape):
    """
    Pick the most likely shooter from a list of person bounding boxes.

    Heuristic:
      - Filter to boxes overlapping the middle 60% horizontally
      - Score = area * vertical_position_weight
        (vertical weight favors lower-middle of frame, where players are)
      - Return the highest-scoring box, or None if no boxes pass

    Args:
        boxes: list of (x1, y1, x2, y2, conf) tuples (pixel coords)
        frame_shape: (h, w)
    """
    h, w = frame_shape[:2]
    if not boxes:
        return None

    middle_x_min = w * 0.20
    middle_x_max = w * 0.80

    best = None
    best_score = -1
    for x1, y1, x2, y2, conf in boxes:
        cx = (x1 + x2) / 2
        if cx < middle_x_min or cx > middle_x_max:
            continue
        cy = (y1 + y2) / 2
        area = max(0, x2 - x1) * max(0, y2 - y1)

        # Vertical weight: peak around 0.6 (lower-middle), drop off at top and very bottom
        vy = cy / h
        v_weight = 1.0 - abs(vy - 0.6) * 1.2  # peaks at vy=0.6
        v_weight = max(0.1, v_weight)

        score = area * v_weight * conf
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)
    return best


def detect_persons(yolo_model, frame):
    """Return list of (x1, y1, x2, y2, conf) for class 'person'."""
    results = yolo_model.predict(frame, classes=[0], verbose=False, conf=0.25)
    boxes = []
    if not results:
        return boxes
    r = results[0]
    if r.boxes is None:
        return boxes
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf[0].cpu().numpy())
        boxes.append((x1, y1, x2, y2, conf))
    return boxes


def expand_box(box, frame_shape, pad=0.15):
    """Expand a bounding box by `pad` fraction on each side, clipped to frame."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    return x1, y1, x2, y2


def keypoints_from_crop(landmarks, crop_box, frame_shape):
    """
    MediaPipe returns landmarks normalized to the cropped image.
    Convert them back to be normalized to the original frame, so the rest of
    the pipeline (which expects [0,1] frame-relative coords) still works.
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = crop_box
    cw = x2 - x1
    ch = y2 - y1
    out = np.zeros((N_LANDMARKS, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        # lm.x, lm.y are in [0,1] of the crop. Convert to crop pixel, then to frame fraction.
        px = x1 + lm.x * cw
        py = y1 + lm.y * ch
        out[i, 0] = px / w
        out[i, 1] = py / h
        out[i, 2] = lm.z
        out[i, 3] = lm.visibility
    return out


def extract_pose_for_clip(clip_path, yolo_model, pose, n_frames=N_FRAMES):
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
    last_valid = None

    for i, idx in enumerate(target_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            if last_valid is not None:
                keypoints[i] = last_valid
            continue

        # Step 1: detect people, pick best one
        boxes = detect_persons(yolo_model, frame)
        best = pick_best_person(boxes, frame.shape)
        if best is None:
            if last_valid is not None:
                keypoints[i] = last_valid
            continue

        # Step 2: expand and crop
        crop_box = expand_box(best, frame.shape, pad=0.15)
        x1, y1, x2, y2 = crop_box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            if last_valid is not None:
                keypoints[i] = last_valid
            continue

        # Step 3: run MediaPipe pose on the crop
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            arr = keypoints_from_crop(result.pose_landmarks.landmark, crop_box, frame.shape)
            keypoints[i] = arr
            last_valid = arr
            detected_mask[i] = True
        else:
            if last_valid is not None:
                keypoints[i] = last_valid

    cap.release()
    return keypoints, detected_mask, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--yolo_model", default="yolov8n.pt",
                        help="YOLO model name (yolov8n.pt is small/fast, yolov8s.pt is more accurate)")
    args = parser.parse_args()

    if not args.clips_dir.exists():
        print(f"ERROR: clips dir not found at {args.clips_dir.resolve()}")
        sys.exit(1)
    labels_path = args.labels or (args.clips_dir / "labels.csv")
    df = pd.read_csv(labels_path)
    print(f"Found {len(df)} clips")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO (downloads weights to ~/.cache on first run)
    print("Loading YOLOv8...")
    from ultralytics import YOLO
    yolo_model = YOLO(args.yolo_model)
    print(f"Loaded {args.yolo_model}")

    # Init MediaPipe pose with higher complexity for better accuracy
    pose = mp.solutions.pose.Pose(
        static_image_mode=True,           # process each crop independently
        model_complexity=2,               # most accurate variant
        enable_segmentation=False,
        min_detection_confidence=0.3,
    )

    n_done = 0
    n_skipped = 0
    failures = []
    low_detection = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pose v2"):
        filename = row["filename"]
        clip_path = args.clips_dir / filename
        if not clip_path.exists():
            failures.append((filename, "missing"))
            continue
        out_path = args.out_dir / f"{Path(filename).stem}.npz"
        if out_path.exists() and not args.force:
            n_skipped += 1
            continue

        keypoints, detected_mask, err = extract_pose_for_clip(clip_path, yolo_model, pose)
        if err:
            failures.append((filename, err))
            continue
        n_detected = int(detected_mask.sum())
        if n_detected < 4:
            low_detection.append((filename, n_detected))
        np.savez_compressed(out_path,
                            keypoints=keypoints,
                            detected_mask=detected_mask)
        n_done += 1

    pose.close()

    print(f"\nProcessed {n_done} clips, skipped (already done) {n_skipped}")
    if failures:
        print(f"\nFailures: {len(failures)}")
        for f, err in failures[:10]:
            print(f"  {f}: {err}")
    if low_detection:
        print(f"\nLow-detection clips (<4 frames): {len(low_detection)}")
        for f, n in low_detection[:10]:
            print(f"  {f}: only {n}/16 frames")
    print(f"\nPose v2 saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
