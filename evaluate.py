"""
Evaluate a trained SwinDAF / DAF checkpoint on a CHAOS dataset split.

Computes per-organ Dice (classes 1-4: Liver, Right Kidney, Left Kidney, Spleen)
in two ways:
  - 2D per-slice Dice: mean +/- std of the Dice score computed independently
    on every 2D slice in the split.
  - 3D per-subject Dice: slices are grouped back into per-subject volumes
    (using the "Subj_<id>slice_<n>.png" filename convention written by
    prepare_data.py) and Dice is computed on the reconstructed 3D volume.

Writes a single JSON file to outputs/test_metrics_<model>_<timestamp>.json
so that README tables are always regenerated from a real run, never typed
by hand.

Usage:
    python evaluate.py --weights model/Best_SwinDAF-CHAOS.pth --split test
    python evaluate.py --weights model/Best_DAF-baseline.pth --model daf --split test
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import dc
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import analytics  # noqa: E402
from common.utils import getTargetSegmentation  # noqa: E402
from data import medicalDataLoader  # noqa: E402
from models.my_stacked_danet import DAF_stack  # noqa: E402
from models.swin_danet import SwinDAF  # noqa: E402

ORGAN_NAMES = {1: "Liver", 2: "Right Kidney", 3: "Left Kidney", 4: "Spleen"}
NUM_CLASSES = 5
# Same thresholds src/demo.py uses to flag a region for human review.
CONFIDENCE_THRESHOLD = 0.5
ENTROPY_THRESHOLD = 1.0
IMG_SIZE = 224
FILENAME_RE = re.compile(r"Subj_(\d+)slice_(\d+)\.png")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, swin_encoder: str) -> torch.nn.Module:
    if model_name == "swin_daf":
        return SwinDAF(num_classes=NUM_CLASSES, encoder_name=swin_encoder, pretrained=False)
    return DAF_stack()


def load_checkpoint(net: torch.nn.Module, weights_path: Path, device: torch.device) -> None:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
        # Checkpoints saved with pickle_module=dill can still fail the
        # weights_only restricted unpickler; fall back to a full load.
        state = torch.load(weights_path, map_location=device, weights_only=False)
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as err:
        raise RuntimeError(
            f"Failed to load checkpoint '{weights_path}' into the model. "
            "Check that --model / --swin_encoder match how this checkpoint was trained."
        ) from err


def parse_subject_slice(img_path: str) -> tuple[str, int] | None:
    match = FILENAME_RE.search(Path(img_path).name)
    if not match:
        return None
    subj_id, slice_num = match.groups()
    return subj_id, int(slice_num)


def dice_2d(pred_mask: np.ndarray, gt_mask: np.ndarray, cls: int, eps: float = 1e-8) -> float:
    pred_c = pred_mask == cls
    gt_c = gt_mask == cls
    inter = np.logical_and(pred_c, gt_c).sum()
    denom = pred_c.sum() + gt_c.sum()
    return float((2 * inter + eps) / (denom + eps))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, type=str, help="Path to a .pth checkpoint")
    parser.add_argument("--root", default="DataSet", type=str, help="Dataset root (contains train/val/test)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], type=str)
    parser.add_argument("--model", default="swin_daf", choices=["daf", "swin_daf"], type=str)
    parser.add_argument(
        "--swin_encoder",
        default="swin_tiny_patch4_window7_224",
        choices=[
            "swin_tiny_patch4_window7_224",
            "swin_small_patch4_window7_224",
            "swin_base_patch4_window7_224",
        ],
        type=str,
    )
    parser.add_argument("--num_workers", default=0, type=int, help="0 is safest on Windows")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--out_dir", default="outputs", type=str)
    parser.add_argument(
        "--no_log_db", action="store_true",
        help="Skip logging per-slice results to outputs/analytics.db",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    dataset = medicalDataLoader.MedicalImageDataset(
        args.split, args.root, transform=transform, mask_transform=mask_transform,
        augment=False, equalize=False,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    print(f" Split: {args.split}  ({len(dataset)} slices)")

    net = build_model(args.model, args.swin_encoder)
    load_checkpoint(net, Path(args.weights), device)
    net.to(device).eval()

    # class -> list of per-slice Dice scores
    dice_2d_scores: dict[int, list[float]] = defaultdict(list)
    # subject_id -> class -> list of (slice_num, mask_2d) for 3D reconstruction
    pred_volumes: dict[str, dict[int, list[tuple[int, np.ndarray]]]] = defaultdict(lambda: defaultdict(list))
    gt_volumes: dict[str, dict[int, list[tuple[int, np.ndarray]]]] = defaultdict(lambda: defaultdict(list))

    db_path = REPO_ROOT / args.out_dir / "analytics.db"

    with torch.no_grad():
        for image, labels, img_paths in loader:
            t0 = time.perf_counter()
            image = image.to(device)
            logits = net(image)
            probs = F.softmax(logits, dim=1)
            pred_mask = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            latency = time.perf_counter() - t0

            # Reuse the exact same GT decoding as training (common.utils.getTargetSegmentation)
            # so class thresholds can never drift between train/eval.
            gt_mask = getTargetSegmentation(labels).cpu().numpy().astype(np.uint8)
            gt_mask = np.clip(gt_mask, 0, NUM_CLASSES - 1)

            slice_dice: dict[int, float] = {}
            for cls in ORGAN_NAMES:
                slice_dice[cls] = dice_2d(pred_mask, gt_mask, cls)
                dice_2d_scores[cls].append(slice_dice[cls])

            parsed = parse_subject_slice(img_paths[0])
            if parsed is not None:
                subj_id, slice_num = parsed
                for cls in ORGAN_NAMES:
                    pred_volumes[subj_id][cls].append((slice_num, pred_mask == cls))
                    gt_volumes[subj_id][cls].append((slice_num, gt_mask == cls))

            if not args.no_log_db:
                probs_np = probs.squeeze(0).cpu().numpy()  # (5, H, W)
                entropy_map = -(probs_np * np.log(probs_np + 1e-8)).sum(axis=0)
                total_px = pred_mask.size
                organ_stats = {}
                for cls, name in ORGAN_NAMES.items():
                    organ_mask = pred_mask == cls
                    count = int(organ_mask.sum())
                    stats = {"pixels": count, "pct": count / total_px * 100}
                    if count > 0:
                        mean_conf = float(probs_np[cls][organ_mask].mean())
                        mean_ent = float(entropy_map[organ_mask].mean())
                        stats["mean_confidence"] = mean_conf
                        stats["mean_entropy"] = mean_ent
                        stats["low_confidence"] = (
                            mean_conf < CONFIDENCE_THRESHOLD or mean_ent > ENTROPY_THRESHOLD
                        )
                    else:
                        stats["low_confidence"] = None
                    organ_stats[name] = stats

                analytics.log_inference(
                    db_path,
                    slice_id=Path(img_paths[0]).stem,
                    source="evaluate",
                    model_name=args.model,
                    encoder_name=args.swin_encoder if args.model == "swin_daf" else None,
                    latency_seconds=latency,
                    organ_stats=organ_stats,
                    dice_scores={name: slice_dice[cls] for cls, name in ORGAN_NAMES.items()},
                )

    # ── 2D per-slice summary ──────────────────────────────────────────────
    per_organ_2d = {}
    for cls, name in ORGAN_NAMES.items():
        scores = np.array(dice_2d_scores[cls])
        per_organ_2d[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
    mean_dice_2d = float(np.mean([v["mean"] for v in per_organ_2d.values()]))

    # ── 3D per-subject summary ────────────────────────────────────────────
    per_organ_3d = {}
    subjects = sorted(pred_volumes.keys())
    for cls, name in ORGAN_NAMES.items():
        subj_scores = []
        for subj_id in subjects:
            pred_slices = sorted(pred_volumes[subj_id][cls], key=lambda t: t[0])
            gt_slices = sorted(gt_volumes[subj_id][cls], key=lambda t: t[0])
            pred_vol = np.stack([m for _, m in pred_slices], axis=0)
            gt_vol = np.stack([m for _, m in gt_slices], axis=0)
            if pred_vol.sum() == 0 and gt_vol.sum() == 0:
                subj_scores.append(1.0)
                continue
            subj_scores.append(float(dc(pred_vol, gt_vol)))
        subj_scores = np.array(subj_scores)
        per_organ_3d[name] = {
            "mean": float(subj_scores.mean()),
            "std": float(subj_scores.std()),
            "n_subjects": len(subj_scores),
        }
    mean_dice_3d = float(np.mean([v["mean"] for v in per_organ_3d.values()])) if subjects else None

    print("\n Per-organ 2D Dice (mean +/- std):")
    for name, v in per_organ_2d.items():
        print(f"   {name:<14} {v['mean']:.4f} +/- {v['std']:.4f}")
    print(f"   {'Mean':<14} {mean_dice_2d:.4f}")

    if subjects:
        print("\n Per-organ 3D Dice (mean +/- std, per-subject):")
        for name, v in per_organ_3d.items():
            print(f"   {name:<14} {v['mean']:.4f} +/- {v['std']:.4f}  (n={v['n_subjects']})")
        print(f"   {'Mean':<14} {mean_dice_3d:.4f}")
    else:
        print("\n No 'Subj_<id>slice_<n>' filenames found — skipping 3D per-subject Dice.")

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"test_metrics_{args.model}_{timestamp}.json"

    result = {
        "checkpoint": str(Path(args.weights)),
        "model": args.model,
        "swin_encoder": args.swin_encoder if args.model == "swin_daf" else None,
        "split": args.split,
        "dataset_root": args.root,
        "num_slices": len(dataset),
        "num_subjects_3d": len(subjects),
        "seed": args.seed,
        "date_utc": timestamp,
        "device": str(device),
        "dice_2d_per_organ": per_organ_2d,
        "mean_dice_2d": mean_dice_2d,
        "dice_3d_per_organ": per_organ_3d if subjects else None,
        "mean_dice_3d": mean_dice_3d,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
