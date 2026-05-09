# Basketball Shot Type Classification

CMSC426 final project. Comparing three approaches — frame-based, video-based, and pose-based — for classifying basketball shot types from short broadcast clips.

## Setup

```bash
# From the parent CMSC426 folder
cv_project_env\Scripts\activate

# Install requirements (one-time)
pip install -r final_project/requirements.txt
```

## Dataset

Clips live outside the repo at `../Final Project NBA dataset/labeled/`. Each clip is labeled with a shot type (3-pointer, jumpshot, layup, dunk) and outcome (make/miss).

Class counts (current):
- three_pointer: ~60
- jumpshot: ~60
- layup: ~60
- dunk: ~25

## Project structure

```
final_project/
├── data/splits/        # train/val/test split CSVs
├── src/                # scripts
├── results/            # checkpoints, plots, metrics
└── notebooks/          # exploration
```

## Workflow

1. `python src/make_split.py` — generate stratified train/val/test split
2. `python src/extract_frames.py` — extract middle frames from clips
3. `python src/visualize_samples.py` — sanity check the dataset
4. (next: training scripts)

## Authors

Tyler [+ partner name]
