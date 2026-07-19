"""Correctness tests for the Dice metric and CHAOS mask-value <-> class mapping."""

from __future__ import annotations

import torch

from common.utils import (
    DicesToDice,
    computeDiceOneHot,
    getOneHotSegmentation,
    getTargetSegmentation,
)

# Normalized pixel values for raw mask levels {0, 63, 126, 189, 252} once loaded
# through ToTensor() (which divides by 255) - classes 0..4.
_LEVELS = [0.0, 63 / 255, 126 / 255, 189 / 255, 252 / 255]


def test_dice_identical_masks_is_one():
    dicer = computeDiceOneHot()
    x = torch.ones(4, 4)

    assert dicer.dice(x, x).item() == 1.0


def test_dice_disjoint_masks_is_zero():
    dicer = computeDiceOneHot()
    pred = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    assert dicer.dice(pred, target).item() == 0.0


def test_dice_known_partial_overlap():
    dicer = computeDiceOneHot()
    # 1 pixel of overlap out of 2 "on" pixels each -> 2*1 / (2+2) = 0.5
    pred = torch.tensor([1.0, 1.0, 0.0, 0.0])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    assert dicer.dice(pred, target).item() == 0.5


def test_dice_both_empty_masks_is_one_not_undefined():
    dicer = computeDiceOneHot()
    x = torch.zeros(4, 4)

    assert dicer.dice(x, x).item() == 1.0


def test_dices_to_dice_aggregates_across_batch():
    # Two samples: one perfect (inter=4, sum=8), one disjoint (inter=0, sum=8)
    dices = torch.tensor([[4.0, 8.0], [0.0, 8.0]])

    result = DicesToDice(dices).item()

    # (2*4 + eps) / (16 + eps) == 0.5
    assert abs(result - 0.5) < 1e-4


def test_compute_dice_one_hot_forward_identical_batches_scores_near_one():
    dicer = computeDiceOneHot()
    onehot = torch.zeros(1, 5, 4, 4)
    onehot[:, 1] = 1.0  # every pixel labeled "liver"

    dices = dicer(onehot, onehot)
    for cls_dice in dices:
        assert abs(DicesToDice(cls_dice).item() - 1.0) < 1e-4


def test_get_target_segmentation_maps_normalized_levels_to_class_indices():
    batch = torch.tensor(_LEVELS).view(1, 1, 1, 5)

    classes = getTargetSegmentation(batch)

    assert classes.tolist() == [0, 1, 2, 3, 4]


def test_get_one_hot_segmentation_matches_target_segmentation():
    batch = torch.tensor(_LEVELS).view(1, 1, 1, 5)

    one_hot = getOneHotSegmentation(batch)  # (1, 5, 1, 5)
    classes = getTargetSegmentation(batch)

    for i, cls in enumerate(classes.tolist()):
        picked = one_hot[0, :, 0, i].tolist()
        assert picked[cls] == 1.0
        assert sum(picked) == 1.0
