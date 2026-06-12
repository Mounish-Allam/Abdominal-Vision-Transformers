"""
Prepare the CHAOS MRI dataset for training.

Downloads from Kaggle (if not already cached), reads T2SPIR DICOM slices,
converts them to PNG, and organises everything into:

    DataSet/
      train/  Img/  GT/
      val/    Img/  GT/
      test/   Img/  GT/

File naming: Subj_<subject_id>slice_<slice_num>.png

Usage:
    python prepare_data.py
    python prepare_data.py --out_dir MyDataset --val_ratio 0.15 --test_ratio 0.05
"""

import os
import argparse
import random
import shutil
import numpy as np
from PIL import Image
import pydicom
import kagglehub

KAGGLE_DATASET = "omarxadel/chaos-combined-ct-mr-healthy-abdominal-organ"
MR_SEQUENCE    = "T2SPIR"   # best sequence for abdominal organ segmentation


def load_dicom_as_png(dcm_path: str) -> Image.Image:
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    # Normalise to 0-255
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def prepare(raw_root: str, out_dir: str, val_ratio: float, test_ratio: float, seed: int):
    random.seed(seed)

    mr_train = os.path.join(raw_root, "CHAOS_Train_Sets", "Train_Sets", "MR")
    subjects  = sorted(os.listdir(mr_train))

    random.shuffle(subjects)
    n_val   = max(1, int(len(subjects) * val_ratio))
    n_test  = max(1, int(len(subjects) * test_ratio))
    n_train = len(subjects) - n_val - n_test

    splits = {
        "train": subjects[:n_train],
        "val":   subjects[n_train : n_train + n_val],
        "test":  subjects[n_train + n_val :],
    }

    print(f"Subjects — train:{len(splits['train'])}  val:{len(splits['val'])}  test:{len(splits['test'])}")

    for split, subj_list in splits.items():
        img_dir = os.path.join(out_dir, split, "Img")
        gt_dir  = os.path.join(out_dir, split, "GT")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(gt_dir,  exist_ok=True)

        for subj in subj_list:
            subj_path = os.path.join(mr_train, subj, MR_SEQUENCE)
            if not os.path.isdir(subj_path):
                print(f"  [skip] {subj} — {MR_SEQUENCE} not found")
                continue

            dicom_dir = os.path.join(subj_path, "DICOM_anon")
            gt_src    = os.path.join(subj_path, "Ground")

            if not os.path.isdir(dicom_dir) or not os.path.isdir(gt_src):
                print(f"  [skip] {subj} — missing DICOM_anon or Ground folder")
                continue

            dcm_files = sorted([f for f in os.listdir(dicom_dir) if f.endswith(".dcm")])
            gt_files  = sorted([f for f in os.listdir(gt_src)    if f.endswith(".png")])

            if len(dcm_files) != len(gt_files):
                print(f"  [warn] {subj} — {len(dcm_files)} DICOMs vs {len(gt_files)} GTs, using min")

            n_slices = min(len(dcm_files), len(gt_files))
            for i in range(n_slices):
                fname = f"Subj_{subj}slice_{i+1}.png"

                # Convert DICOM → PNG
                img = load_dicom_as_png(os.path.join(dicom_dir, dcm_files[i]))
                img.save(os.path.join(img_dir, fname))

                # Copy GT mask
                shutil.copy(os.path.join(gt_src, gt_files[i]), os.path.join(gt_dir, fname))

            print(f"  [{split}] Subject {subj:>4} — {n_slices} slices written")

    total = sum(
        len(os.listdir(os.path.join(out_dir, s, "Img")))
        for s in ["train", "val", "test"]
    )
    print(f"\nDone. {total} total slices saved to '{out_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",    default="DataSet",  help="Output dataset folder")
    parser.add_argument("--val_ratio",  default=0.10, type=float)
    parser.add_argument("--test_ratio", default=0.10, type=float)
    parser.add_argument("--seed",       default=42,   type=int)
    args = parser.parse_args()

    print("Fetching CHAOS dataset from Kaggle cache ...")
    raw_root = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Dataset at: {raw_root}\n")

    prepare(raw_root, args.out_dir, args.val_ratio, args.test_ratio, args.seed)
