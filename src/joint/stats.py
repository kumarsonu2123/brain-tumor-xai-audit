import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def patient_bootstrap_indices(patient_ids, n_boot=1000, seed=42):
    # each draw: patients with replacement, every drawn patient brings all of its rows
    pids = np.asarray(patient_ids)
    groups = [np.flatnonzero(pids == p) for p in np.unique(pids)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]


def bootstrap_ci(fn, arrays, boot_indices, alpha=0.05):
    vals = np.array([fn(*(a[idx] for a in arrays)) for idx in boot_indices], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))]


def excludes_zero(ci, null=0.0):
    return bool(np.isfinite(ci).all() and (ci[0] > null or ci[1] < null))


def spearman_nan(x, y, min_n=8):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n or x[ok].std() == 0 or y[ok].std() == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok])[0])


def auroc_nan(score, positive):
    s, p = np.asarray(score, float), np.asarray(positive, float)
    ok = np.isfinite(s) & np.isfinite(p)
    p = p[ok].astype(bool)
    if p.all() or not p.any():
        return float("nan")
    return float(roc_auc_score(p, s[ok]))


def mean_nan(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")