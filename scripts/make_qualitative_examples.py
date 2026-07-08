"""
Generate real qualitative example images (input / ground-truth / prediction)
from the trained SwinDAF checkpoint on actual CHAOS test slices.

Also prints a per-subject, per-organ Dice breakdown -- useful for picking
which slices are genuinely representative of "good" and "bad" cases, and for
grounding the README's failure analysis in real per-subject numbers rather
than assumptions.

Usage:
    python scripts/make_qualitative_examples.py --weights model/Best_SwinDAF-CHAOS.pth
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from common.utils import getTargetSegmentation  # noqa: E402
from data import medicalDataLoader  # noqa: E402
from models.swin_danet import SwinDAF  # noqa: E402

ORGAN_NAMES = {1: "Liver", 2: "Right Kidney", 3: "Left Kidney", 4: "Spleen"}
OVERLAY_RGBA = {
    1: (255, 60, 60, 180),
    2: (60, 220, 60, 180),
    3: (60, 120, 255, 180),
    4: (255, 210, 0, 180),
}
IMG_SIZE = 224
FILENAME_RE = re.compile(r"Subj_(\d+)slice_(\d+)\.png")


def dice_2d(pred_mask, gt_mask, cls, eps=1e-8):
    pred_c = pred_mask == cls
    gt_c = gt_mask == cls
    inter = np.logical_and(pred_c, gt_c).sum()
    denom = pred_c.sum() + gt_c.sum()
    return float((2 * inter + eps) / (denom + eps))


def make_overlay(gray_uint8: np.ndarray, mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    base = Image.fromarray(gray_uint8).convert("RGBA")
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id, rgba in OVERLAY_RGBA.items():
        layer[mask == cls_id] = rgba
    return Image.alpha_composite(base, Image.fromarray(layer, "RGBA")).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--root", default="DataSet", type=str)
    parser.add_argument("--out_dir", default="outputs/qualitative", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SwinDAF(num_classes=5, encoder_name="swin_tiny_patch4_window7_224", pretrained=False)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    net.load_state_dict(state, strict=True)
    net.to(device).eval()

    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    mask_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = medicalDataLoader.MedicalImageDataset(
        "test", args.root, transform=transform, mask_transform=mask_transform, augment=False, equalize=False,
    )

    per_subject_organ_scores = defaultdict(lambda: defaultdict(list))
    slice_records = []  # (subj_id, slice_num, img_path, mean_dice, per_organ_dice, gray, pred_mask, gt_mask)

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, labels, img_path = dataset[idx]
            match = FILENAME_RE.search(Path(img_path).name)
            if not match:
                continue
            subj_id, slice_num = match.groups()

            logits = net(image.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)
            pred_mask = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            gt_mask = getTargetSegmentation(labels.unsqueeze(0)).cpu().numpy().astype(np.uint8)
            gt_mask = np.clip(gt_mask, 0, 4)

            organ_dice = {name: dice_2d(pred_mask, gt_mask, cls) for cls, name in ORGAN_NAMES.items()}
            for name, d in organ_dice.items():
                per_subject_organ_scores[subj_id][name].append(d)

            gray = (image[0].numpy() * 255).astype(np.uint8)
            slice_records.append((subj_id, int(slice_num), img_path, np.mean(list(organ_dice.values())), organ_dice, gray, pred_mask, gt_mask))

    print("\nPer-subject, per-organ mean 2D Dice (test split):")
    for subj_id in sorted(per_subject_organ_scores):
        print(f"  Subject {subj_id}:")
        for name in ORGAN_NAMES.values():
            scores = per_subject_organ_scores[subj_id][name]
            print(f"    {name:<14} {np.mean(scores):.4f}  (n={len(scores)} slices)")

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exclude slices with no organs at all in the ground truth from best/median/worst
    # selection -- an empty-GT slice trivially scores a "perfect" Dice by predicting
    # nothing, which is a real but misleading artifact of the metric, not a genuine
    # demonstration of the model working.
    n_total = len(slice_records)
    non_trivial = [r for r in slice_records if (r[7] > 0).any()]
    n_excluded = n_total - len(non_trivial)
    if n_excluded:
        print(f"\nExcluded {n_excluded}/{n_total} slices with an empty ground truth "
              f"(no organs present) from best/median/worst selection.")

    non_trivial.sort(key=lambda r: r[3])
    worst = non_trivial[0]
    best = non_trivial[-1]
    median = non_trivial[len(non_trivial) // 2]

    for tag, rec in [("worst", worst), ("median", median), ("best", best)]:
        subj_id, slice_num, img_path, mean_dice, organ_dice, gray, pred_mask, gt_mask = rec
        gt_overlay = make_overlay(gray, gt_mask)
        pred_overlay = make_overlay(gray, pred_mask)
        combined = Image.new("RGB", (IMG_SIZE * 3 + 20, IMG_SIZE + 30), "white")
        combined.paste(Image.fromarray(gray).convert("RGB"), (0, 20))
        combined.paste(gt_overlay, (IMG_SIZE + 10, 20))
        combined.paste(pred_overlay, (IMG_SIZE * 2 + 20, 20))
        fname = f"{tag}_Subj{subj_id}_slice{slice_num}_meanDice{mean_dice:.2f}.png"
        combined.save(out_dir / fname)
        print(f"\n[{tag}] Subject {subj_id} slice {slice_num} -- mean Dice {mean_dice:.3f}")
        for name, d in organ_dice.items():
            print(f"    {name:<14} {d:.4f}")
        print(f"    saved: {out_dir / fname}")


if __name__ == "__main__":
    main()
