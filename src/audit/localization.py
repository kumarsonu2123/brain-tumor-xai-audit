import numpy as np


def _mask(mask):
    return np.asarray(mask).astype(bool)


def iou_dice(heatmap, mask, threshold=0.5):
    m = _mask(mask)
    if not m.any():
        return float("nan"), float("nan")
    b = np.asarray(heatmap) >= threshold
    inter = np.logical_and(b, m).sum()
    iou = inter / np.logical_or(b, m).sum()
    dice = 2 * inter / (b.sum() + m.sum())
    return float(iou), float(dice)


def pointing_game(heatmap, mask):
    m = _mask(mask)
    if not m.any():
        return float("nan")
    h = np.asarray(heatmap)
    peak = h == h.max()
    # ties (clipped or flat maps) count as a hit only if most of the tied peak is inside
    return float(m[peak].mean() >= 0.5)


def energy_in_mask(heatmap, mask):
    m = _mask(mask)
    h = np.asarray(heatmap, dtype=np.float64)
    assert h.min() >= 0, "energy needs a non-negative map"
    total = h.sum()
    if not m.any() or total <= 0:
        return float("nan")
    return float(h[m].sum() / total)


def mask_fraction(mask):
    return float(_mask(mask).mean())