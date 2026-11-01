import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def entropy(p, axis=-1):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def mc_decomposition(samples):
    # samples: (T, N, C) softmax outputs
    s = np.asarray(samples, dtype=np.float64)
    mean = s.mean(axis=0)
    pred = mean.argmax(axis=1)
    pe = entropy(mean)
    ee = entropy(s).mean(axis=0)
    return {
        "mean_probs": mean,
        "mc_pred": pred,
        "mc_mean_confidence": mean.max(axis=1),
        "predictive_entropy": pe,
        "expected_entropy": ee,
        "mutual_information": np.clip(pe - ee, 0.0, None),   # >= 0 by Jensen; clip float noise
        "predicted_class_std": s[:, np.arange(s.shape[1]), pred].std(axis=0),
    }


def reliability_bins(confidence, correct, n_bins=15):
    conf, corr = np.asarray(confidence, float), np.asarray(correct, float)
    idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)   # bins (lo, hi]
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"lo": b / n_bins, "hi": (b + 1) / n_bins, "count": int(m.sum()),
                     "accuracy": float(corr[m].mean()) if m.any() else float("nan"),
                     "confidence": float(conf[m].mean()) if m.any() else float("nan")})
    return rows


def ece(confidence, correct, n_bins=15):
    rows = reliability_bins(confidence, correct, n_bins)
    n = sum(r["count"] for r in rows)
    return float(sum(r["count"] * abs(r["accuracy"] - r["confidence"]) for r in rows if r["count"]) / n)


def nll(probs, y):
    p = np.asarray(probs, float)[np.arange(len(y)), np.asarray(y)]
    return float(-np.log(np.clip(p, EPS, 1.0)).mean())


def brier(probs, y):
    p = np.asarray(probs, float)
    onehot = np.eye(p.shape[1])[np.asarray(y)]
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def auroc(score, is_error):
    e = np.asarray(is_error).astype(bool)
    if e.all() or not e.any():
        return float("nan")   # undefined without both classes (e.g. a bootstrap draw with no errors)
    return float(roc_auc_score(e, score))


def patient_bootstrap_indices(patient_ids, n_boot=1000, seed=42):
    # each draw: patients with replacement, all of a drawn patient's rows together
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