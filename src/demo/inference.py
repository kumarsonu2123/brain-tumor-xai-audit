import numpy as np
import torch
from matplotlib import colormaps
from torchvision import tv_tensors

from src.data.dataset import minmax_uint8
from src.data.transforms import get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.explain import gradcam, integrated_gradients
from src.models.classifier import load_checkpoint
from src.uncertainty.mc_dropout import assert_mc_mode, enable_mc_dropout
from src.uncertainty.metrics import mc_decomposition

DISCLAIMER = "Research demo, not a medical device. Do not use for diagnosis."


def to_gray2d(image):
    a = np.asarray(image)
    if a.ndim == 3:                     # RGB(A) upload -> grey; the model only ever saw grey slices
        a = a[..., :3].astype(np.float32).mean(axis=2)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D slice or an RGB image, got shape {a.shape}")
    return a


def overlay(gray01, heat, alpha=0.45):
    rgb = np.repeat(gray01[..., None], 3, axis=2)
    hm = colormaps["jet"](np.clip(heat, 0, 1))[..., :3]
    return (np.clip((1 - alpha) * rgb + alpha * hm, 0, 1) * 255).astype(np.uint8)


class TumorExplainer:
    def __init__(self, ckpt_path, thresholds, device=None, seed=42):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ckpt = load_checkpoint(ckpt_path, self.device)
        self.class_names = ckpt["class_names"]
        self.image_size = ckpt["config"]["image_size"]
        self.tf = get_eval_transforms(self.image_size)
        self.th = thresholds
        self.T = int(thresholds["n_forward_passes"])
        self.ig_steps = int(thresholds["ig_steps"])
        self.target_layer = thresholds["target_layer"]
        self.seed = seed

    def preprocess(self, image):
        u8 = minmax_uint8(to_gray2d(image))
        x = self.tf(tv_tensors.Image(torch.from_numpy(u8).unsqueeze(0).repeat(3, 1, 1))).as_subclass(torch.Tensor)
        gray01 = (x[0] * IMAGENET_STD[0] + IMAGENET_MEAN[0]).clamp(0, 1).numpy()
        return x.float(), gray01

    @torch.no_grad()
    def _mc(self, x):
        enable_mc_dropout(self.model)
        assert_mc_mode(self.model)
        torch.manual_seed(self.seed)
        # one batch of T copies: each copy draws its own dropout mask; BN is in eval so copies don't interact
        p = self.model(x[None].to(self.device).repeat(self.T, 1, 1, 1)).float().softmax(1).cpu().numpy()
        self.model.eval()
        return p[:, None, :]            # (T, N=1, C)

    def __call__(self, image):
        x, gray01 = self.preprocess(image)
        self.model.eval()
        with torch.no_grad():
            det = self.model(x[None].to(self.device)).float().softmax(1)[0].cpu().numpy()
        target = int(det.argmax())
        samples = self._mc(x)
        d = mc_decomposition(samples)
        mc_probs = d["mean_probs"][0]
        gc = gradcam.explain(self.model, x, target=target, target_layer=self.target_layer)
        ig = integrated_gradients.explain(self.model, x, target=target, n_steps=self.ig_steps)

        conf = float(mc_probs[target])
        ent = float(d["predictive_entropy"][0])
        area = float((gc >= self.th["gradcam_area_threshold"]).mean())
        reasons = []
        if conf < self.th["min_confidence"]:
            reasons.append(f"MC confidence {conf:.3f} < {self.th['min_confidence']:.3f}")
        if ent > self.th["max_entropy"]:
            reasons.append(f"predictive entropy {ent:.3f} > {self.th['max_entropy']:.3f}")
        if self.th.get("max_gradcam_area") is not None and area > self.th["max_gradcam_area"]:
            reasons.append(f"diffuse Grad-CAM ({area:.1%} of image > {self.th['max_gradcam_area']:.1%})")
        return {
            "predicted_class": self.class_names[target],
            "class_index": target,
            "mc_confidence": conf,
            "mc_probs": {c: float(p) for c, p in zip(self.class_names, mc_probs)},
            "det_probs": {c: float(p) for c, p in zip(self.class_names, det)},
            "mc_predicted_class": self.class_names[int(mc_probs.argmax())],
            "predictive_entropy": ent,
            "mutual_information": float(d["mutual_information"][0]),
            "gradcam_area": area,
            "trust_flag": "LOW TRUST - review" if reasons else "no flag raised",
            "flag_reasons": reasons,
            "gradcam": gc, "ig": ig, "image": gray01,
            "gradcam_overlay": overlay(gray01, gc), "ig_overlay": overlay(gray01, ig),
        }