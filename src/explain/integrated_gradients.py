import numpy as np
import torch
from captum.attr import IntegratedGradients

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.explain._common import as_batch, predicted_class


def black_baseline(x):
    # black in pixel space == -mean/std in the normalised space the model actually sees
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((0.0 - mean) / std).expand_as(x).contiguous()


def explain(model, image_tensor, target=None, n_steps=50, internal_batch_size=50,
            clip_percentile=99.0, return_raw=False):
    x = as_batch(model, image_tensor)
    if target is None:
        target = predicted_class(model, x)

    ig = IntegratedGradients(model)
    attr, delta = ig.attribute(x, baselines=black_baseline(x), target=int(target), n_steps=n_steps,
                               internal_batch_size=internal_batch_size, return_convergence_delta=True)
    signed = attr[0].sum(dim=0).detach().float().cpu().numpy()   # (H, W)

    mag = np.abs(signed)
    scale = np.percentile(mag, clip_percentile)
    heat = np.clip(mag / scale, 0.0, 1.0).astype(np.float32) if scale > 0 else np.zeros_like(mag)

    if return_raw:
        return heat, {"signed": signed.astype(np.float32), "delta": float(delta.item()), "target": int(target)}
    return heat