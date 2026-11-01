import numpy as np
import pytest
import torch
from torch import nn
from torchvision.transforms import v2

from src.audit import faithfulness as F
from src.audit import localization as L
from src.audit import stability as S


def square_mask(n=224, lo=80, hi=150):
    m = np.zeros((n, n), np.uint8)
    m[lo:hi, lo:hi] = 1
    return m


# ---------------- localization ----------------

def test_perfect_map_localizes():
    m = square_mask()
    h = m.astype(np.float32)
    assert L.iou_dice(h, m) == (1.0, 1.0)
    assert L.pointing_game(h, m) == 1.0
    assert L.energy_in_mask(h, m) == pytest.approx(1.0)


def test_inverted_map_scores_zero():
    m = square_mask()
    h = 1.0 - m.astype(np.float32)
    assert L.iou_dice(h, m) == (0.0, 0.0)
    assert L.pointing_game(h, m) == 0.0
    assert L.energy_in_mask(h, m) == pytest.approx(0.0)


def test_random_map_is_at_chance():
    m = square_mask()
    frac = L.mask_fraction(m)
    rng = np.random.default_rng(0)
    energies, hits = [], []
    for _ in range(300):
        h = rng.random(m.shape)
        energies.append(L.energy_in_mask(h, m))
        hits.append(L.pointing_game(h, m))
    assert np.mean(energies) == pytest.approx(frac, abs=0.01)
    assert np.mean(hits) == pytest.approx(frac, abs=0.06)
    iou, _ = L.iou_dice(rng.random(m.shape), m)
    assert iou < 0.2


def test_empty_mask_is_nan():
    m = np.zeros((32, 32), np.uint8)
    h = np.random.default_rng(0).random((32, 32))
    assert all(np.isnan(v) for v in [*L.iou_dice(h, m), L.pointing_game(h, m), L.energy_in_mask(h, m)])


def test_tied_peak_majority_rule():
    m = np.zeros((10, 10), np.uint8); m[:, :3] = 1
    flat = np.ones((10, 10))                       # every pixel is the peak, 30% inside -> miss
    assert L.pointing_game(flat, m) == 0.0
    h = np.zeros((10, 10)); h[:, :2] = 1; h[:, 9] = 1   # 20 of 30 tied peak pixels inside -> hit
    assert L.pointing_game(h, m) == 1.0


# ---------------- faithfulness ----------------

class RegionModel(nn.Module):
    # logits [s, -s] with s = 2 * (mean region intensity - 0.5): p(class 0) goes 0.02 -> 0.98 roughly
    # in proportion to how much of the region is present (a steeper gain saturates and hides differences)
    def __init__(self, region):
        super().__init__()
        self.register_buffer("region", torch.as_tensor(region, dtype=torch.bool))
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        s = 2 * (x[:, :, self.region].mean(dim=(1, 2)) - 0.5) + self.dummy
        return torch.stack([s, -s], dim=1)


def _toy():
    region = np.zeros((32, 32), bool); region[4:12, 4:12] = True
    model = RegionModel(region).eval()
    image = torch.ones(3, 32, 32)
    zeros = torch.zeros(3, 32, 32)
    return model, image, zeros, region


def test_auc_known_curves():
    assert F.auc(np.ones(21)) == pytest.approx(1.0)
    assert F.auc(np.linspace(1, 0, 21)) == pytest.approx(0.5)


def test_step_masks_monotone():
    order = F.pixel_order(np.random.default_rng(1).random((8, 8)))
    m = F.step_masks(order, steps=8)
    assert m.shape == (9, 64) and not m[0].any() and m[-1].all()
    assert (np.diff(m.sum(1)) > 0).all()
    assert (m[:-1] <= m[1:]).all()          # once perturbed, stays perturbed


def test_tie_break_is_seeded():
    flat = np.zeros((8, 8))
    assert (F.pixel_order(flat, 3) == F.pixel_order(flat, 3)).all()
    assert not (F.pixel_order(flat, 3) == F.pixel_order(flat, 4)).all()
    assert not (F.pixel_order(flat, 3) == np.arange(64)).all()   # not raster order


def test_perfect_explainer_beats_random_deletion_and_insertion():
    model, image, zeros, region = _toy()
    perfect = region.astype(np.float32)
    rand = np.random.default_rng(0).random((32, 32))
    kw = dict(target=0, steps=16, start=zeros)
    del_perfect = F.auc(F.perturbation_curve(model, image, perfect, mode="deletion", **kw))
    del_random = F.auc(F.perturbation_curve(model, image, rand, mode="deletion", **kw))
    ins_perfect = F.auc(F.perturbation_curve(model, image, perfect, mode="insertion", **kw))
    ins_random = F.auc(F.perturbation_curve(model, image, rand, mode="insertion", **kw))
    assert del_perfect < del_random - 0.1
    assert ins_perfect > ins_random + 0.1


def test_curve_endpoints():
    model, image, zeros, region = _toy()
    h = np.random.default_rng(0).random((32, 32))
    d = F.perturbation_curve(model, image, h, target=0, mode="deletion", steps=8, start=zeros)
    i = F.perturbation_curve(model, image, h, target=0, mode="insertion", steps=8, start=zeros)
    full, empty = torch.softmax(model(image[None]), 1)[0, 0].item(), torch.softmax(model(zeros[None]), 1)[0, 0].item()
    assert d[0] == pytest.approx(full) and d[-1] == pytest.approx(empty)
    assert i[0] == pytest.approx(empty) and i[-1] == pytest.approx(full)


# ---------------- stability ----------------

def _blob(n=224, cy=80, cx=140, s=6):
    yy, xx = np.mgrid[:n, :n]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2)).astype(np.float32)


def test_identical_maps_similarity_one():
    h = _blob()
    assert S.spearman_similarity(h, h) == pytest.approx(1.0)
    assert S.ssim_similarity(h, h) == pytest.approx(1.0)


def test_constant_map_is_nan():
    assert np.isnan(S.spearman_similarity(np.zeros((16, 16)), _blob(16, 8, 8, 3)))
    assert np.isnan(S.ssim_similarity(np.zeros((16, 16)), _blob(16, 8, 8, 3)))


def _smooth_field(n=224, sigma=4, seed=0):
    # structure everywhere: a mostly-zero map would let SSIM's background (~1 where both are 0) hide misalignment
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(np.random.default_rng(seed).random((n, n)), sigma)
    return ((f - f.min()) / (f.max() - f.min())).astype(np.float32)


@pytest.mark.parametrize("deg", [5, -5])
def test_rotation_alignment_matches_pipeline(deg):
    h = _smooth_field()
    spec = {"type": "rotate", "degrees": deg}
    rotated = v2.RandomRotation(degrees=(deg, deg))(torch.from_numpy(h)[None])[0].numpy()  # same op as get_stability_transform
    valid = S.valid_region(h.shape, spec)
    aligned = S.align_to_original(rotated, spec)
    assert 0.9 < valid.mean() < 1.0
    assert S.spearman_similarity(h, aligned, valid) > 0.95
    assert S.ssim_similarity(h, aligned, valid) > 0.9
    assert S.ssim_similarity(h, rotated, valid) < S.ssim_similarity(h, aligned, valid) - 0.2
    assert S.spearman_similarity(h, rotated, valid) < S.spearman_similarity(h, aligned, valid) - 0.2
    wrong_way = S.align_to_original(rotated, {"type": "rotate", "degrees": -deg})
    assert S.ssim_similarity(h, wrong_way, valid) < S.ssim_similarity(h, aligned, valid) - 0.2


def test_brightness_needs_no_alignment():
    h = _blob()
    spec = {"type": "brightness", "factor": 1.1}
    assert (S.align_to_original(h, spec) == h).all()
    assert S.valid_region(h.shape, spec).all()