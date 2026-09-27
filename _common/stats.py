"""Small, dependency-light statistics helpers. All randomness takes an explicit seed."""
import numpy as np
from scipy import stats as _st


def rng(seed=0):
    return np.random.default_rng(seed)


def bootstrap_ci(values_or_fn, data=None, n=5000, alpha=0.05, seed=0):
    """bootstrap_ci(array) -> (mean, lo, hi) of the mean.
    bootstrap_ci(fn, data) -> (fn(data), lo, hi) resampling rows of `data` (array or list of arrays with same len)."""
    r = rng(seed)
    if data is None:
        x = np.asarray(values_or_fn, dtype=float)
        x = x[~np.isnan(x)]
        idx = r.integers(0, len(x), size=(n, len(x)))
        bs = x[idx].mean(axis=1)
        return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))
    fn = values_or_fn
    arrs = data if isinstance(data, (list, tuple)) else [data]
    arrs = [np.asarray(a) for a in arrs]
    N = len(arrs[0])
    est = fn(*arrs)
    bs = []
    for _ in range(n):
        i = r.integers(0, N, size=N)
        bs.append(fn(*[a[i] for a in arrs]))
    bs = np.asarray(bs, dtype=float)
    return float(est), float(np.nanquantile(bs, alpha / 2)), float(np.nanquantile(bs, 1 - alpha / 2))


def wilson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"),) * 3
    z = _st.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def ece(conf, correct, bins=10):
    """Top-label ECE with equal-width bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def brier_multi(P, y):
    """P: (N,K) probs, y: (N,) int labels."""
    P = np.asarray(P, float)
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), np.asarray(y, int)] = 1
    return float(((P - Y) ** 2).sum(axis=1).mean())


def logloss(P, y, eps=1e-6):
    P = np.clip(np.asarray(P, float), eps, 1)
    P = P / P.sum(axis=1, keepdims=True)
    return float(-np.log(P[np.arange(len(y)), np.asarray(y, int)]).mean())


def corr(x, y, method="spearman"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if method == "spearman":
        r, p = _st.spearmanr(x[m], y[m])
    else:
        r, p = _st.pearsonr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def corr_ci(x, y, method="spearman", n=5000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    f = (lambda a, b: _st.spearmanr(a, b)[0]) if method == "spearman" else (lambda a, b: _st.pearsonr(a, b)[0])
    return bootstrap_ci(f, [x[m], y[m]], n=n, seed=seed)


def mcnemar(a_correct, b_correct):
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    p = _st.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return {"a_only": n01, "b_only": n10, "p": float(p)}


def paired_perm_test(diff, n=10000, seed=0):
    """Sign-flip permutation test for mean(diff)=0. Returns two-sided p."""
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    r = rng(seed)
    obs = abs(d.mean())
    signs = r.choice([-1, 1], size=(n, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-12).sum()) / (n + 1))


def kl(p, q, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1)
    q = np.clip(np.asarray(q, float), eps, 1)
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    return (p * np.log(p / q)).sum(-1)


def js(p, q, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1)
    q = np.clip(np.asarray(q, float), eps, 1)
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    m = (p + q) / 2
    return 0.5 * (p * np.log(p / m)).sum(-1) + 0.5 * (q * np.log(q / m)).sum(-1)
