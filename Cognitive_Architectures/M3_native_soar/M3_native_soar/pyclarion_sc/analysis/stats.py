"""Statistics used in the papers: t-tests, Cohen's d, 2x2 mixed ANOVA, correlations, slopes."""
from __future__ import annotations

import numpy as np
from scipy import stats


def describe(x) -> dict:
    x = np.asarray(x, float)
    return {"M": float(x.mean()), "SD": float(x.std(ddof=1)), "n": int(len(x))}


def cohen_d_pooled(a, b) -> float:
    """d with the average of the two SDs (matches the values in Myrseth et al.)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def paired_t(a, b) -> dict:
    r = stats.ttest_rel(a, b)
    return {"t": float(r.statistic), "df": len(a) - 1, "p": float(r.pvalue), "d": cohen_d_pooled(a, b)}


def independent_t(a, b) -> dict:
    r = stats.ttest_ind(a, b)
    return {"t": float(r.statistic), "df": len(a) + len(b) - 2, "p": float(r.pvalue),
            "d": cohen_d_pooled(a, b)}


def mixed_anova_interaction(g1_a, g1_b, g2_a, g2_b) -> dict:
    """
    2 (between: group) x 2 (within: measure) interaction.
    For a 2x2 mixed design, F_interaction = t^2 of an independent t-test on the
    within-subject difference scores.
    """
    d1 = np.asarray(g1_a, float) - np.asarray(g1_b, float)
    d2 = np.asarray(g2_a, float) - np.asarray(g2_b, float)
    r = stats.ttest_ind(d1, d2)
    return {"F": float(r.statistic ** 2), "df": (1, len(d1) + len(d2) - 2), "p": float(r.pvalue)}


def pearson(x, y) -> dict:
    r = stats.pearsonr(x, y)
    return {"r": float(r.statistic), "p": float(r.pvalue)}


def ols(y, X, names) -> dict:
    """Ordinary least squares with standard errors. X must include an intercept column."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - X.shape[1]
    sigma2 = resid @ resid / df
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    return {"beta": dict(zip(names, beta)), "se": dict(zip(names, se)), "t": dict(zip(names, t)),
            "p": dict(zip(names, p)), "cov": cov, "names": list(names), "df": df}


def fmt_p(p: float) -> str:
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".")
