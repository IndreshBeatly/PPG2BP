import os
import random
from pathlib import Path

# Config
FINAL_DATA_DIR = 'data/final'
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# Output split lists (you can use these later to load data)
TRAIN_LIST_PATH = 'split_train.txt'
VAL_LIST_PATH = 'split_val.txt'
TEST_LIST_PATH = 'split_test.txt'

# Step 1: Gather all patients that have a signals_with_metadata.npz file
patient_dirs = [
    p.name for p in Path(FINAL_DATA_DIR).iterdir()
    if p.is_dir() and (p / "signals_with_metadata.npz").exists()
]

print(f"[INFO] Found {len(patient_dirs)} valid patients with processed .npz files.")

# Step 2: Shuffle and split
random.seed(SEED)
random.shuffle(patient_dirs)

total = len(patient_dirs)
train_end = int(TRAIN_SPLIT * total)
val_end = train_end + int(VAL_SPLIT * total)

train_ids = patient_dirs[:train_end]
val_ids = patient_dirs[train_end:val_end]
test_ids = patient_dirs[val_end:]

# Step 3: Save splits
def save_split(ids, path):
    with open(path, 'w') as f:
        for pid in ids:
            f.write(f"{pid}\n")

save_split(train_ids, TRAIN_LIST_PATH)
save_split(val_ids, VAL_LIST_PATH)
save_split(test_ids, TEST_LIST_PATH)

# Final report
print(f"[DONE] Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
print(f"→ Saved split files: {TRAIN_LIST_PATH}, {VAL_LIST_PATH}, {TEST_LIST_PATH}")
