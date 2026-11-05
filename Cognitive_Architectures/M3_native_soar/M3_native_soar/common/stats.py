"""
Shared statistics (Document 1, Section 9), copied unchanged from soar-clarion/experiments/analysis.py.

    paired_diffs(df, variant, a, b, measure, scenario)  per-seed difference between two conditions
    test_diff(x)        paired t-test on differences x, with mean, 95% bootstrap CI, d_z and Wilcoxon p
    tost(x, lo, hi)     equivalence test (two one-sided tests)
    noninferior(x, m)   one-sided test that the mean difference is below a margin m
    holm(results)       Holm correction of the p-values in a family of tests
    h1b_diffs(df, v)    per-seed anticipation effect: (after - before door) in anticipation minus in constant
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20261101)
B = 10_000


def boot_ci(x, fn=np.mean):
    x = np.asarray(x, float)
    if len(x) < 2 or np.all(x == x[0]):
        v = fn(x) if len(x) else float("nan")
        return [float(v), float(v)]
    idx = RNG.integers(0, len(x), (B, len(x)))
    vals = np.array([fn(x[i]) for i in idx]) if fn is not np.mean else x[idx].mean(axis=1)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def dz(x):
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else (0.0 if x.mean() == 0 else float("inf") * np.sign(x.mean()))


def paired_diffs(df, variant, a, b, measure, scenario):
    """Per-seed difference measure(cond a) - measure(cond b) for one variant."""
    d = df[(df.variant == variant) & (df.scenario == scenario)]
    A = d[d.condition == a].set_index("seed")[measure]
    Bv = d[d.condition == b].set_index("seed")[measure]
    j = pd.concat([A, Bv], axis=1, keys=["a", "b"]).dropna()
    return (j["a"] - j["b"]).to_numpy()


def test_diff(x, label):
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return {"label": label, "n": n, "mean": float("nan"), "p": float("nan")}
    if np.all(x == x[0]):
        t, p = (float("nan"), 1.0) if x[0] == 0 else (float("inf"), 0.0)
        w = p
    else:
        t, p = stats.ttest_1samp(x, 0.0)
        try:
            w = stats.wilcoxon(x).pvalue
        except ValueError:
            w = float("nan")
    return {"label": label, "n": n, "mean": float(x.mean()), "ci95": boot_ci(x), "dz": dz(x),
            "dz_ci95": boot_ci(x, dz) if not np.all(x == x[0]) else [dz(x), dz(x)],
            "t": float(t), "p": float(p), "wilcoxon_p": float(w)}


def tost(x, lo, hi, label):
    """Paired TOST: H0 mean <= lo or mean >= hi. Equivalent if both one-sided p < .05."""
    x = np.asarray(x, float)
    n, m = len(x), float(x.mean())
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    if se == 0:
        p_lo = 0.0 if m > lo else 1.0
        p_hi = 0.0 if m < hi else 1.0
    else:
        p_lo = 1 - stats.t.cdf((m - lo) / se, n - 1)
        p_hi = stats.t.cdf((m - hi) / se, n - 1)
    p = max(p_lo, p_hi)
    return {"label": label, "n": n, "mean": m, "ci90": [m - stats.t.ppf(.95, max(n - 1, 1)) * se,
                                                        m + stats.t.ppf(.95, max(n - 1, 1)) * se],
            "bounds": [lo, hi], "p": float(p), "equivalent": bool(p < .05)}


def noninferior(x, margin, label):
    """H0: mean >= margin (i.e. at least `margin` higher). Supported if p < .05."""
    x = np.asarray(x, float)
    n, m = len(x), float(x.mean())
    se = x.std(ddof=1) / np.sqrt(n)
    p = (0.0 if m < margin else 1.0) if se == 0 else float(stats.t.cdf((m - margin) / se, n - 1))
    return {"label": label, "n": n, "mean": m, "ci95": boot_ci(x), "margin": margin, "p": p}


def holm(results):
    ps = [r["p"] for r in results]
    order = np.argsort(ps)
    adj = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    for r, a in zip(results, adj):
        r["p_holm"] = float(a)
    return results


def h1b_diffs(df, variant):
    d = df[(df.variant == variant) & (df.scenario == "anticipation")]
    a = d[d.condition == "anticipation"].set_index("seed")
    c = d[d.condition == "constant"].set_index("seed")
    j = pd.concat([a.lapse_past - a.lapse_approach, c.lapse_past - c.lapse_approach], axis=1, keys=["a", "c"]).dropna()
    return (j.a - j.c).to_numpy()


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def fci(c, d=3):
    return f"[{fmt(c[0], d)}, {fmt(c[1], d)}]" if c else "[n/a]"


def fp(p):
    return "n/a" if p is None or np.isnan(p) else ("< .001" if p < .001 else f"{p:.3f}")
