import numpy as np
import torch
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F


def spec_name(spec):
    if spec["type"] == "rotate":
        return f"rotate{spec['degrees']:+g}"
    return f"brightness_x{spec['factor']:g}"


def _rotate(arr, degrees, interpolation):
    t = torch.as_tensor(np.asarray(arr, dtype=np.float32))[None]
    return F.rotate(t, angle=float(degrees), interpolation=interpolation, fill=0.0)[0].numpy()


def align_to_original(heatmap, spec):
    # undo the geometric part of the perturbation so both maps live in the original image frame
    if spec["type"] == "rotate":
        return _rotate(heatmap, -spec["degrees"], InterpolationMode.BILINEAR)
    return np.asarray(heatmap, dtype=np.float32)


def valid_region(shape, spec):
    if spec["type"] != "rotate":
        return np.ones(shape, dtype=bool)
    # nearest forward (as the image pipeline does), bilinear back (as align_to_original does)
    fwd = _rotate(np.ones(shape, np.float32), spec["degrees"], InterpolationMode.NEAREST)
    return _rotate(fwd, -spec["degrees"], InterpolationMode.BILINEAR) > 0.999


def spearman_similarity(a, b, valid=None):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if valid is not None:
        a, b = a[valid], b[valid]
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def ssim_similarity(a, b, valid=None):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    _, smap = structural_similarity(a, b, data_range=1.0, full=True)
    return float(smap[valid].mean() if valid is not None else smap.mean())