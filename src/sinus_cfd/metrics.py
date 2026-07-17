"""
Segmentation overlap metrics shared across evaluation scripts.

Kept deliberately small and dependency-light (numpy only) so both the
classical-baseline evaluation (scripts/evaluate_nasalseg_dice.py) and the
nnU-Net-vs-classical comparison (scripts/compare_nnunet_vs_classical.py)
compute Dice the *same* way — otherwise the two numbers aren't comparable.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def dice_coefficient(pred: np.ndarray, truth: np.ndarray) -> float:
    """
    Dice overlap of two binary masks: 2|A∩B| / (|A|+|B|).

    Two empty masks are defined to agree perfectly (1.0) rather than 0/0.
    """
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    denom = int(pred.sum()) + int(truth.sum())
    if denom == 0:
        return 1.0
    intersection = int((pred & truth).sum())
    return 2.0 * intersection / denom


def labels_to_mask(label_volume: np.ndarray, keep: Iterable[int]) -> np.ndarray:
    """Binary mask of voxels whose label is in ``keep``."""
    return np.isin(label_volume, list(int(v) for v in keep))


def per_label_dice(
    pred_labels: np.ndarray,
    truth_labels: np.ndarray,
    label_ids: Iterable[int],
) -> dict[int, float]:
    """Dice for each label id treated as its own binary mask."""
    return {
        int(lid): dice_coefficient(pred_labels == lid, truth_labels == lid)
        for lid in label_ids
    }
