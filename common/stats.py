"""Statistics helpers so every reported number carries uncertainty.

Design choices worth knowing:
  * The unit of resampling is the QUESTION. Claims from one answer are correlated, so a Wilson
    interval over pooled claims would be too narrow. `bootstrap_ci` and `paired_bootstrap` take one
    value per question.
  * System comparisons are PAIRED (same questions, both systems), which is far more sensitive than
    comparing two independent intervals.
  * With several comparisons against one reference, p-values are Holm-adjusted.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n. Use only for independent trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _percentile(sorted_xs: Sequence[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    pos = q * (len(sorted_xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def bootstrap_ci(xs: Sequence[float], iters: int = 5000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean with a percentile bootstrap CI. Returns {"mean","lo","hi","n"}; NaN fields when empty."""
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(rng.choices(xs, k=n)) / n for _ in range(iters))
    return {"mean": sum(xs) / n, "lo": _percentile(means, alpha / 2), "hi": _percentile(means, 1 - alpha / 2), "n": n}


def ratio_bootstrap_ci(num: Sequence[float], den: Sequence[float], iters: int = 5000, seed: int = 0) -> dict:
    """CI for sum(num)/sum(den) resampling questions, e.g. pooled unsupported-claim rate where each
    question contributes (unsupported, total) claims. Questions with den=0 still count as resamplable."""
    pairs = list(zip(num, den))
    tot = sum(d for _, d in pairs)
    if not pairs or tot == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": len(pairs)}
    rng = random.Random(seed)
    stats = []
    for _ in range(iters):
        s = rng.choices(pairs, k=len(pairs))
        d = sum(b for _, b in s)
        if d:
            stats.append(sum(a for a, _ in s) / d)
    stats.sort()
    return {"mean": sum(a for a, _ in pairs) / tot, "lo": _percentile(stats, .025), "hi": _percentile(stats, .975),
            "n": len(pairs)}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], iters: int = 5000, seed: int = 0) -> dict:
    """Mean difference a-b over paired items with a bootstrap CI and a two-sided sign-flip
    permutation p-value. Items where either side is None/NaN are dropped pairwise."""
    d = [x - y for x, y in zip(a, b) if x is not None and y is not None and not (math.isnan(x) or math.isnan(y))]
    if not d:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n": 0}
    ci = bootstrap_ci(d, iters, seed)
    return {"diff": ci["mean"], "lo": ci["lo"], "hi": ci["hi"], "p": sign_flip_p(d, iters, seed), "n": len(d)}


def sign_flip_p(diffs: Sequence[float], iters: int = 5000, seed: int = 0) -> float:
    """Two-sided paired permutation test of mean(diffs) == 0 (exact when n <= 16)."""
    n = len(diffs)
    if n == 0:
        return float("nan")
    obs = abs(sum(diffs))
    if obs == 0:
        return 1.0
    if n <= 16:
        hits = 0
        for mask in range(1 << n):
            s = sum(-x if mask >> i & 1 else x for i, x in enumerate(diffs))
            hits += abs(s) >= obs - 1e-12
        return hits / (1 << n)
    rng = random.Random(seed)
    hits = sum(abs(sum(x if rng.random() < .5 else -x for x in diffs)) >= obs - 1e-12 for _ in range(iters))
    return (hits + 1) / (iters + 1)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (NaNs pass through)."""
    items = sorted(((p, k) for k, p in pvals.items() if not math.isnan(p)))
    m, out, running = len(items), {k: p for k, p in pvals.items() if math.isnan(p)}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Agreement between two raters beyond chance (1 = perfect, 0 = chance)."""
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    labels = set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((list(a).count(c) / n) * (list(b).count(c) / n) for c in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def percentile(xs: Sequence[float], q: float) -> float:
    return _percentile(sorted(xs), q)
