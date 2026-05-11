# Basketball Shot Type Classification

CMSC426 final project. Comparing three approaches — frame-based, video-based, and pose-based — for classifying basketball shot types from short broadcast clips.

**Authors:** Tyler [Last Name], [Partner Name]

## Results summary

| Approach | Backbone | Test Accuracy | Macro F1 |
|---|---|---|---|
| Frame | ResNet-50 (ImageNet) | 0.558 | 0.504 |
| Video | R(2+1)D-18 (Kinetics-400) | **0.628** | **0.566** |
| Pose v1 | MediaPipe + BiLSTM | 0.209 | 0.157 |
| Pose v2 | YOLOv8 + MediaPipe + BiLSTM | 0.256 | 0.212 |

Dataset: 214 NBA broadcast clips labeled across four shot types (dunk, jumpshot, layup, three-pointer). 60/20/20 stratified train/val/test split (121/41/41 clips after dropping 2 free-throw outliers).

---

## Setup

### Prerequisites
- **Windows 10/11** (Linux/Mac should work but commands below use Windows paths)
- **Python 3.11** ([download here](https://www.python.org/downloads/release/python-3119/))
- **NVIDIA GPU with at least 4 GB VRAM** and recent drivers (CUDA 12.x)
- **Git** ([download here](https://git-scm.com/download/win))
- **ffmpeg** on PATH ([builds here](https://www.gyan.dev/ffmpeg/builds/), only needed if you'll re-trim clips)
- **~3 GB free disk space** for environment + dependencies + extracted frames

### Clone the repo

```
git clone https://github.com/YOUR_USERNAME/cmsc426-shot-classification.git
cd cmsc426-shot-classification
```

### Create a virtual environment

From the parent folder (the one containing `final_project/`):

```
py -3.11 -m venv cv_project_env
cv_project_env\Scripts\activate
```

You should see `(cv_project_env)` in your prompt. **You must reactivate the venv every time you open a new terminal:**

```
cv_project_env\Scripts\activate
```

### Install PyTorch with CUDA

Critical: use the exact command below. The default `pip install torch` will give you a CPU-only build that won't use your GPU.

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### Install everything else

```
cd final_project
pip install -r requirements.txt
pip install ultralytics tensorboard
```

### Verify GPU is detected

```
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

You should see `CUDA: True` and your GPU model. If `CUDA: False`, your PyTorch install is wrong — uninstall and redo the install step above.

---

## Dataset

The 214 labeled clips are **not included in the repo** (too large for Git). To run the pipeline, you need them in this exact structure:

```
<parent folder>/
├── Final Project NBA dataset/
│   └── labeled/
│       ├── dunk_001.mp4
│       ├── dunk_002.mp4
│       ├── ...
│       ├── three_pointer_060.mp4
│       └── labels.csv
└── cmsc426-shot-classification/      <- this repo
    └── final_project/
```

`labels.csv` should have columns: `filename`, `original_filename`, `shot_type`, `outcome`, `notes`.

Get the dataset from Tyler (Google Drive / shared folder). Place the `Final Project NBA dataset/` folder one level above the repo folder.

---

## Reproducing the results

All commands run from the `final_project/` directory with the venv activated. Each step writes its outputs to a subfolder and is idempotent (safe to rerun).

### Step 1 — Generate the train/val/test split

```
python src\make_split.py
```

Writes `data/splits/train.csv`, `val.csv`, `test.csv`. **Only run this once.** The split is committed to the repo so all team members use the same one. Rerunning produces the same split (fixed random seed).

### Step 2 — Extract frames for the frame model

```
python src\extract_frames.py
```

Extracts 3 frames per clip (at 25%, 50%, 75%). Writes JPEGs to `data/frames/<clip_name>/frame_0.jpg`, `frame_1.jpg`, `frame_2.jpg`. Takes ~1 minute. Idempotent.

### Step 3 — Train and evaluate the frame model

```
python src\train_frame_model.py --frame_mode random --run_name frame_baseline_3frames
python src\eval_frame_model.py --run_name frame_baseline_3frames
```

Training: ~5 minutes on a GTX 1070 Ti. The eval script prints test accuracy, per-class F1, and saves:
- `results/runs/frame_baseline_3frames/best_model.pth` — checkpoint
- `results/runs/frame_baseline_3frames/confusion_matrix_test.png` — figure
- `results/runs/frame_baseline_3frames/predictions_test.csv` — per-clip predictions
- `results/runs/frame_baseline_3frames/summary_test.json` — clean metrics

### Step 4 — Extract frames for the video model

```
python src\extract_video_frames.py
```

Extracts 16 frames per clip for R(2+1)D-18. Writes to `data/video_frames/<clip_name>/frame_00.jpg` through `frame_15.jpg`. Takes ~3 minutes. Idempotent.

### Step 5 — Train and evaluate the video model

```
python src\train_video_model.py --run_name video_baseline
python src\eval_video_model.py --run_name video_baseline
```

Training: ~20 minutes on a GTX 1070 Ti. Uses batch size 4 by default. If you hit out-of-memory errors:

```
python src\train_video_model.py --run_name video_baseline --batch_size 2
```

### Step 6 — Extract pose keypoints

```
python src\extract_pose_v2.py
```

Runs YOLOv8 person detection on each frame, picks the most likely shooter, crops to that person, then runs MediaPipe Pose. Writes `data/pose_v2/<clip_name>.npz` per clip. Takes ~10 minutes (CPU-bound). The first run also downloads `yolov8n.pt` (~6 MB) automatically.

If MediaPipe errors with `AttributeError: module 'mediapipe' has no attribute 'solutions'`, install the legacy version:

```
pip install "mediapipe==0.10.14"
```

### Step 7 — Train and evaluate the pose model

```
python src\train_pose_model.py --pose_dir data/pose_v2 --run_name pose_v2
python src\eval_pose_model.py --pose_dir data/pose_v2 --run_name pose_v2
```

Training: ~2 minutes. Pose model performs near chance (~25% accuracy) — see the report for analysis of why.

---

## Watching training live (optional)

In a **second terminal** (also venv-activated), from `final_project/`:

```
tensorboard --logdir results/runs
```

Open http://localhost:6006. You'll see live loss/accuracy curves for all runs, useful for comparing approaches side-by-side.

---

## Sanity-check tools

These don't affect training but help verify the pipeline is working correctly.

| Script | Purpose |
|---|---|
| `src/visualize_samples.py` | 4×4 grid of training frames with labels |
| `src/visualize_triplets.py` | Show all 3 extracted frames per clip side-by-side |
| `src/visualize_pose.py` | Overlay MediaPipe skeleton on video frames |

Example:
```
python src\visualize_samples.py
python src\visualize_triplets.py --class_filter jumpshot
python src\visualize_pose.py --n 6
```

---

## Run everything from scratch

If you want to reproduce all results from a fresh clone, run these commands in order:

```
python src\make_split.py
python src\extract_frames.py
python src\extract_video_frames.py
python src\extract_pose_v2.py

python src\train_frame_model.py --frame_mode random --run_name frame_baseline_3frames
python src\eval_frame_model.py --run_name frame_baseline_3frames

python src\train_video_model.py --run_name video_baseline
python src\eval_video_model.py --run_name video_baseline

python src\train_pose_model.py --pose_dir data/pose_v2 --run_name pose_v2
python src\eval_pose_model.py --pose_dir data/pose_v2 --run_name pose_v2
```

Total wall-clock time on a GTX 1070 Ti: roughly 45 minutes for extraction + 30 minutes for training.

---

## Project structure

```
final_project/
├── data/
│   ├── splits/             # train.csv, val.csv, test.csv (committed)
│   ├── frames/             # 3 frames per clip (not committed, regeneratable)
│   ├── video_frames/       # 16 frames per clip (not committed)
│   └── pose_v2/            # MediaPipe keypoints per clip (not committed)
├── src/
│   ├── make_split.py
│   ├── extract_frames.py
│   ├── extract_video_frames.py
│   ├── extract_pose.py
│   ├── extract_pose_v2.py
│   ├── dataset.py
│   ├── video_dataset.py
│   ├── pose_dataset.py
│   ├── train_frame_model.py
│   ├── train_video_model.py
│   ├── train_pose_model.py
│   ├── eval_frame_model.py
│   ├── eval_video_model.py
│   ├── eval_pose_model.py
│   ├── visualize_samples.py
│   ├── visualize_triplets.py
│   └── visualize_pose.py
├── results/
│   └── runs/
│       ├── frame_baseline_3frames/
│       ├── video_baseline/
│       └── pose_v2/
├── requirements.txt
└── README.md               # this file
```

---

## Troubleshooting

**`python: can't open file '...': [Errno 2] No such file or directory`**
You're running the script from the wrong folder. `cd` into `final_project/` first.

**`ERROR: labels.csv not found`**
The dataset isn't in the expected location. Check that `Final Project NBA dataset/labeled/labels.csv` exists one level above the repo folder.

**`CUDA out of memory`**
Drop the batch size. For the video model: `--batch_size 2`. For the frame model: `--batch_size 8`.

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
Newer MediaPipe removed the legacy API. Install the older version: `pip install "mediapipe==0.10.14"`.

**Training is very slow (~minutes per epoch on the frame model)**
You're running on CPU. Reinstall PyTorch with CUDA support (see Setup → Install PyTorch with CUDA).

**`OSError: [WinError 127] ... c10_cuda.dll`**
PyTorch's CUDA runtime DLL won't load. Install the Visual C++ Redistributable for VS 2015–2022 (x64) from https://aka.ms/vs/17/release/vc_redist.x64.exe, restart your terminal, and try again.
