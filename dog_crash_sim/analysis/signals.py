"""Signal processing used by crash-test assessment (SAE J211 / ISO 6487 style)."""

from __future__ import annotations

import math

import numpy as np


def cfc_filter(x: np.ndarray, dt: float, cfc: float) -> np.ndarray:
    """SAE J211 channel frequency class filter (4-pole phaseless Butterworth).

    A 2-pole Butterworth is applied forward and then backward along axis 0.
    """
    x = np.asarray(x, float)
    if len(x) < 4:
        return x.copy()
    wd = 2.0 * math.pi * cfc * 2.0775
    wa = math.sin(wd * dt / 2.0) / math.cos(wd * dt / 2.0)
    den = 1.0 + math.sqrt(2.0) * wa + wa * wa
    a0 = wa * wa / den
    a1 = 2.0 * a0
    a2 = a0
    b1 = -2.0 * (wa * wa - 1.0) / den
    b2 = (-1.0 + math.sqrt(2.0) * wa - wa * wa) / den

    def run(sig):
        y = np.empty_like(sig)
        y[0] = sig[0]
        y[1] = sig[1]
        for i in range(2, len(sig)):
            y[i] = a0 * sig[i] + a1 * sig[i - 1] + a2 * sig[i - 2] + b1 * y[i - 1] + b2 * y[i - 2]
        return y

    flat = x.reshape(len(x), -1)
    out = np.empty_like(flat)
    for k in range(flat.shape[1]):
        # pad with the end values to limit start-up transients
        s = flat[:, k]
        pad = min(len(s) - 1, 100)
        sp = np.concatenate([np.full(pad, s[0]), s, np.full(pad, s[-1])])
        f = run(sp)
        f = run(f[::-1])[::-1]
        out[:, k] = f[pad:pad + len(s)]
    return out.reshape(x.shape)


def hic(t: np.ndarray, a_g: np.ndarray, window: float = 0.015):
    """Head Injury Criterion of a resultant acceleration in g.  Returns (hic, t1, t2)."""
    t = np.asarray(t, float)
    a = np.maximum(np.asarray(a_g, float), 0.0)
    if len(t) < 3:
        return 0.0, 0.0, 0.0
    dt = float(np.median(np.diff(t)))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(t))])
    w = max(1, int(round(window / dt)))
    best, b1, b2 = 0.0, 0, 0
    n = len(t)
    for lag in range(1, w + 1):
        dT = t[lag:] - t[:-lag]
        avg = (cum[lag:] - cum[:-lag]) / dT
        val = dT * avg ** 2.5
        i = int(np.argmax(val))
        if val[i] > best:
            best, b1, b2 = float(val[i]), i, i + lag
    return best, float(t[b1]), float(t[min(b2, n - 1)])


def clip_3ms(t: np.ndarray, a: np.ndarray, duration: float = 0.003) -> float:
    """Highest level exceeded continuously for ``duration`` (the '3 ms clip')."""
    t = np.asarray(t, float)
    a = np.asarray(a, float)
    if len(t) < 3:
        return float(a.max()) if len(a) else 0.0
    dt = float(np.median(np.diff(t)))
    w = max(1, int(round(duration / dt)))
    if w >= len(a):
        return float(a.min())
    from numpy.lib.stride_tricks import sliding_window_view
    return float(sliding_window_view(a, w + 1).min(axis=1).max())


def peak(x) -> float:
    x = np.asarray(x, float)
    return float(np.abs(x).max()) if x.size else 0.0
