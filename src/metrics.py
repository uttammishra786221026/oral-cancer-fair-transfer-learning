"""Screening metrics, bootstrap confidence intervals, McNemar and DeLong tests."""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def confusion(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def screening_metrics(y, p, thr=0.5):
    """cancer = positive class. p = predicted probability of cancer."""
    y, p = np.asarray(y), np.asarray(p); pred = (p >= thr).astype(int)
    tp, tn, fp, fn = confusion(y, pred)
    sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) and not np.isnan(prec) else 0.0
    return {'accuracy': (tp + tn) / len(y), 'sensitivity': sens, 'specificity': spec, 'precision': prec, 'f1': f1,
            'auc': roc_auc_score(y, p) if len(set(y)) == 2 else np.nan, 'balanced_accuracy': (sens + spec) / 2,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def bootstrap_ci(y, p, thr=0.5, n=2000, seed=0):
    """95 % percentile intervals, resampling test images with replacement (stratified by class)."""
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]; keys = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc']
    B = {k: [] for k in keys}
    for _ in range(n):
        idx = np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]
        m = screening_metrics(y[idx], p[idx], thr)
        for k in keys: B[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in B.items()}


def mcnemar(y, pa, pb, thr=0.5):
    """exact McNemar test on the discordant correct/incorrect pairs of two models."""
    y = np.asarray(y); ca = (np.asarray(pa) >= thr).astype(int) == y; cb = (np.asarray(pb) >= thr).astype(int) == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    pval = 1.0 if b + c == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5))
    return b, c, pval


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2


def delong(y, pa, pb):
    """DeLong test for two correlated ROC AUCs (fast implementation, Sun & Xu 2014)."""
    y = np.asarray(y); order = np.argsort(-y, kind='stable'); y = y[order]
    preds = np.vstack([np.asarray(pa)[order], np.asarray(pb)[order]]); m = int(y.sum()); n = len(y) - m
    tx = np.array([_midrank(r[:m]) for r in preds]); ty = np.array([_midrank(r[m:]) for r in preds]); tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    z = (aucs[0] - aucs[1]) / np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-12))
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


def threshold_for_sensitivity(y, p, target=0.90):
    """largest threshold on the validation set that still reaches the target sensitivity"""
    y, p = np.asarray(y), np.asarray(p); cand = np.sort(np.unique(p))[::-1]
    for t in cand:
        if ((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) >= target: return float(t)
    return float(cand[-1])
