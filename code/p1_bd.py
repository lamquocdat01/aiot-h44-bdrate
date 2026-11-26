"""BD-rate helpers (prereg section 2): standard PCHIP BD on ln(bpp) vs mAP, computability rule, isotonic variant."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import isotonic_regression


def bd_rate(r1, q1, r2, q2):
    lo, hi = max(q1.min(), q2.min()), min(q1.max(), q2.max())
    if hi <= lo: return np.nan
    f1 = PchipInterpolator(q1, np.log(r1)); f2 = PchipInterpolator(q2, np.log(r2))
    return float((np.exp((f2.integrate(lo, hi) - f1.integrate(lo, hi)) / (hi - lo)) - 1) * 100)


def strict(q): return bool(np.all(np.diff(q) > 0))


def nonmono(q): return bool(np.any(np.diff(q) < 0))


def iso_curve(r, q):
    qi = isotonic_regression(np.asarray(q, float), increasing=True)
    keep = np.r_[True, np.diff(qi) > 0]
    return np.asarray(r)[keep], qi[keep]


def computable_std(q1, q2):
    return strict(q1) and strict(q2) and min(q1.max(), q2.max()) > max(q1.min(), q2.min())


def bd_std(r1, q1, r2, q2):
    return bd_rate(r1, q1, r2, q2) if computable_std(q1, q2) else np.nan


def bd_iso(r1, q1, r2, q2):
    a, b = iso_curve(r1, q1); c, d = iso_curve(r2, q2)
    if a.size < 2 or c.size < 2: return np.nan
    return bd_rate(a, b, c, d)
