import numpy as np
import torch
import torch.nn.functional as F

from src.explain._common import as_batch, predicted_class


def _resolve(model, dotted):
    layer = model
    for part in dotted.split("."):
        layer = getattr(layer, part)
    return layer


def explain(model, image_tensor, target=None, target_layer="layer4", return_raw=False):
    x = as_batch(model, image_tensor)
    if target is None:
        target = predicted_class(model, x)
    layer = _resolve(model, target_layer)

    store = {}
    handle = layer.register_forward_hook(lambda mod, inp, out: store.__setitem__("act", out))
    try:
        with torch.enable_grad():
            # detach so the caller's tensor isn't flagged; requires_grad keeps the graph alive even with frozen weights
            x = x.detach().requires_grad_(True)
            logits = model(x)
            grads, = torch.autograd.grad(logits[0, int(target)], store["act"])
    finally:
        handle.remove()

    act = store["act"].detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)            # (1, C, 1, 1)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))    # (1, 1, h, w)
    up = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    peak = up.max()
    heat = (up / peak if peak > 0 else torch.zeros_like(up)).cpu().numpy().astype(np.float32)

    if return_raw:
        return heat, {"cam_lowres": cam[0, 0].cpu().numpy(), "target": int(target)}
    return heat