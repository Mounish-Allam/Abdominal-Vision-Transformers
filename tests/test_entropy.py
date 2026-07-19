"""Entropy/uncertainty correctness tests, via demo.compute_stats' entropy formula."""

from __future__ import annotations

import math

import numpy as np

from demo import compute_stats


def test_uniform_softmax_gives_max_entropy():
    # 5 classes, uniform probability at every pixel -> entropy == log(5) everywhere.
    probs = np.full((5, 4, 4), 0.2, dtype=np.float32)
    mask = np.ones((4, 4), dtype=np.int64)  # every pixel labeled class 1

    stats = compute_stats(mask, probs)

    assert math.isclose(stats[1]["mean_entropy"], math.log(5), rel_tol=1e-3)


def test_one_hot_softmax_gives_near_zero_entropy():
    probs = np.zeros((5, 4, 4), dtype=np.float32)
    probs[1] = 1.0  # fully confident "class 1" everywhere
    mask = np.ones((4, 4), dtype=np.int64)

    stats = compute_stats(mask, probs)

    assert stats[1]["mean_entropy"] < 1e-4


def test_one_hot_softmax_gives_max_confidence():
    probs = np.zeros((5, 4, 4), dtype=np.float32)
    probs[1] = 1.0
    mask = np.ones((4, 4), dtype=np.int64)

    stats = compute_stats(mask, probs)

    assert math.isclose(stats[1]["mean_confidence"], 1.0, rel_tol=1e-4)


def test_uniform_softmax_flags_low_confidence():
    probs = np.full((5, 4, 4), 0.2, dtype=np.float32)
    mask = np.ones((4, 4), dtype=np.int64)

    stats = compute_stats(mask, probs)

    assert stats[1]["low_confidence"] is True
