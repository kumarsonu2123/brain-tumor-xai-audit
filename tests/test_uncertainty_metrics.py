import numpy as np
import pytest
import torch
from torch import nn

from src.uncertainty import mc_dropout as MC
from src.uncertainty import metrics as U


def test_entropy_bounds():
    assert U.entropy(np.full(3, 1 / 3)) == pytest.approx(np.log(3))
    assert U.entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-9)


def test_decomposition_identical_passes_have_zero_mi():
    s = np.tile(np.array([[0.7, 0.2, 0.1]]), (30, 1, 1))     # (T, N=1, C)
    d = U.mc_decomposition(s)
    assert d["mutual_information"][0] == pytest.approx(0.0, abs=1e-9)
    assert d["predictive_entropy"][0] == pytest.approx(d["expected_entropy"][0])
    assert d["predicted_class_std"][0] == pytest.approx(0.0)


def test_disagreeing_confident_passes_are_epistemic():
    s = np.array([[[0.98, 0.01, 0.01]], [[0.01, 0.98, 0.01]]] * 15)   # each pass confident, they disagree
    d = U.mc_decomposition(s)
    assert d["mutual_information"][0] > 0.5
    assert d["expected_entropy"][0] < 0.15
    assert d["mutual_information"][0] <= d["predictive_entropy"][0] + 1e-12


def test_mi_nonnegative_and_bounded_random():
    rng = np.random.default_rng(0)
    s = rng.dirichlet(np.ones(3), size=(30, 200))
    d = U.mc_decomposition(s)
    assert (d["mutual_information"] >= 0).all()
    assert (d["mutual_information"] <= d["predictive_entropy"] + 1e-12).all()


def test_ece_calibrated_vs_overconfident():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.4, 1.0, 200_000)
    calibrated = rng.random(conf.size) < conf
    overconf = rng.random(conf.size) < np.clip(conf - 0.2, 0, 1)
    assert U.ece(conf, calibrated) < 0.01
    assert U.ece(conf, overconf) == pytest.approx(0.2, abs=0.02)


def test_proper_scores():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert U.nll(perfect, y) == pytest.approx(0.0, abs=1e-9)
    assert U.brier(perfect, y) == pytest.approx(0.0)
    assert U.brier(np.full((3, 3), 1 / 3), y) == pytest.approx(2 / 3)


def test_auroc_edges():
    assert U.auroc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]) == 1.0
    assert np.isnan(U.auroc([0.1, 0.2], [0, 0]))


def test_bootstrap_keeps_patients_whole():
    pids = np.array(["a", "a", "a", "b", "c", "c"])
    for idx in U.patient_bootstrap_indices(pids, n_boot=50, seed=1):
        for p in np.unique(pids[idx]):
            assert (pids[idx] == p).sum() % (pids == p).sum() == 0


def _tiny_net():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(), nn.Dropout(0.5), nn.Linear(8, 3))
    net.train()
    with torch.no_grad():                      # give BN non-trivial running stats
        for _ in range(5):
            net(torch.randn(16, 3, 12, 12))
    return net


def test_enable_mc_dropout_keeps_bn_frozen():
    net = _tiny_net()
    n = MC.enable_mc_dropout(net)
    s = MC.assert_mc_mode(net)
    assert n == 1 and s["n_bn"] == 1 and not net.training
    before = MC.bn_snapshot(net)
    x = torch.randn(4, 3, 12, 12)
    with torch.no_grad():
        outs = torch.stack([net(x) for _ in range(20)])
    assert outs.std(0).mean() > 0                  # passes actually differ
    assert MC.bn_unchanged(before, net)


def test_model_train_would_corrupt_bn():
    net = _tiny_net()
    before = MC.bn_snapshot(net)
    net.train()                                     # the mistake MC-dropout must avoid
    with torch.no_grad():
        net(torch.randn(4, 3, 12, 12))
    assert not MC.bn_unchanged(before, net)
    with pytest.raises(AssertionError):
        MC.assert_mc_mode(net)


def test_mc_predict_seeded():
    net = _tiny_net()
    MC.enable_mc_dropout(net)
    loader = [{"image": torch.randn(5, 3, 12, 12)}, {"image": torch.randn(3, 3, 12, 12)}]
    a = MC.mc_predict(net, loader, 10, "cpu", seed=7)
    b = MC.mc_predict(net, loader, 10, "cpu", seed=7)
    c = MC.mc_predict(net, loader, 10, "cpu", seed=8)
    assert a.shape == (10, 8, 3) and np.allclose(a.sum(-1), 1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)