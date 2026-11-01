import math

import numpy as np
import torch
import torchvision.transforms.functional as TF

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def pixel_order(heatmap, seed=0):
    # most important first; seeded tie-break so flat regions aren't removed in raster order
    h = np.asarray(heatmap, dtype=np.float64).ravel()
    tie = np.random.default_rng(seed).random(h.size)
    return np.lexsort((tie, -h))


def step_masks(order, steps):
    # (steps+1, n_pixels) bool; row k marks the pixels perturbed after k of `steps` chunks
    n = order.size
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    counts = np.round(np.linspace(0, n, steps + 1)).astype(np.int64)
    return rank[None, :] < counts[:, None]


def black_baseline(image):
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    return ((0.0 - mean) / std).expand_as(image).contiguous()


def blurred_baseline(image, sigma=10.0):
    k = 2 * math.ceil(3 * sigma) + 1
    return TF.gaussian_blur(image.unsqueeze(0), kernel_size=[k, k], sigma=[sigma, sigma])[0]


@torch.no_grad()
def perturbation_curve(model, image, heatmap, target, mode, steps=20, start=None, seed=0, batch_size=64):
    # image: normalised (C, H, W). deletion: image -> start (baseline); insertion: start (blurred) -> image.
    assert not model.training
    device = next(model.parameters()).device
    image = image.to(device)
    C, H, W = image.shape
    if start is None:
        start = black_baseline(image) if mode == "deletion" else blurred_baseline(image)
    start = start.to(device)
    masks = torch.from_numpy(step_masks(pixel_order(heatmap, seed), steps)).to(device).view(steps + 1, 1, H, W)
    if mode == "deletion":
        batch = torch.where(masks, start[None], image[None])
    elif mode == "insertion":
        batch = torch.where(masks, image[None], start[None])
    else:
        raise ValueError(mode)
    probs = [model(batch[i:i + batch_size]).float().softmax(1)[:, int(target)] for i in range(0, len(batch), batch_size)]
    return torch.cat(probs).cpu().numpy()


def auc(curve):
    # trapezoid rule on x in [0, 1] with equal spacing
    c = np.asarray(curve, dtype=np.float64)
    return float(((c[:-1] + c[1:]) / 2).mean())