#!/usr/bin/env python3
"""
Toy AGN example: changepoint detection on an AGN light curve (ZTF19acnskyy).

Uses Astropy to fetch the light curve from IRSA and ruptures (PELT) for segmentation.
Run from project root:  python scripts/plots_astropy.py
Outputs figures to presentation/img/.

Acknowledgments: Code improved with AI-assisted tools (Cursor IDE). PELT/changepoint
workflow informed by the Introduction to Changepoint Analysis workshop (R. Killick, CASI 2024):
https://github.com/rkillick/intro-changepoint-course/blob/master/IntroCptWorkshop.Rmd
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode

import ruptures as rpt

from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "img"
OUT.mkdir(parents=True, exist_ok=True)

# PELT penalty (BIC-style: pen = log(n) for penalty on number of changepoints).
PELT_PEN = None  # Set to np.log(n) at runtime
# Lighter penalty for "change in mean" only (so PELT can detect more mean shifts).
# Smaller value → more changepoints. Try 2.0–3.0 or 0.5*log(n) if still too few.
PELT_PEN_MEAN = 2.0

# Piecewise harmonic: min fraction of n for each segment (window; like R's win).
# Larger value = longer minimum segment, fewer changepoints (e.g. 0.15 = 15%, 0.20 = 20%).
PIECEWISE_MIN_FRAC = 0.20
# Max number of changepoints to find (binary segmentation).
PIECEWISE_MAX_CPTS = 5 # just for example - is just an arbitrary number
# Period of the harmonic (cos/sin) in days. ~182.6 days ≈ half-year, chosen to match the
# observational cadence (visibility gaps every ~6 months), not the source’s intrinsic
# variability—AGN are not strictly periodic. Converts to time-index: period_days/mean(diff(mjd)).
HARMONIC_PERIOD_DAYS = 182.6

# ZTF19acnskyy (SDSS J133519.91+072807.4)
COORD = SkyCoord("13h35m19.91s", "+07d28m07.4s", frame="icrs")
IRSA_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

COLORS = {"data": "maroon", "event": "#2E86AB", "cpt": "#A23B72", "seg": "#F77F00", "band": "#6A994E"}


def load_agn_lightcurve():
    """Fetch ZTF g-band light curve from IRSA and return (TimeSeries, t, mag, mjd, magerr)."""
    params = urlencode({
        "CIRCLE": f"{COORD.ra.deg} {COORD.dec.deg} {5/3600}",
        "BANDNAME": "g",
        "BAD_CATFLAGS_MASK": 32768,
        "FORMAT": "ipac_table",
    })
    with urlopen(f"{IRSA_URL}?{params}", timeout=30) as r:
        tbl = Table.read(io.BytesIO(r.read()), format="ipac")
    if len(tbl) == 0:
        raise ValueError("No data returned from IRSA for this position.")
    # Use first object if multiple in circle
    if "oid" in tbl.colnames:
        oids, inv = np.unique(tbl["oid"], return_inverse=True)
        best = oids[np.argmax(np.bincount(inv))]
        tbl = tbl[tbl["oid"] == best]
    mjd = np.asarray(tbl["mjd"])
    mag = np.asarray(tbl["mag"])
    magerr = np.asarray(tbl["magerr"]) if "magerr" in tbl.colnames else np.full_like(mag, 0.05)
    order = np.argsort(mjd)
    mjd, mag, magerr = mjd[order], mag[order], magerr[order]
    n = len(mjd)
    t = np.arange(1, n + 1, dtype=float)
    ts = TimeSeries(
        time=Time(mjd, format="mjd"),
        data={"mag": mag * u.mag, "mag_err": magerr * u.mag, "t": t},
    )
    return ts, t, mag, mjd, magerr


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150)
    plt.close()
    print("  ", OUT / name)


# --- 1. Simulated change types (for teaching) ---
def plot_change_types():
    np.random.seed(42)
    x = np.arange(500)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        ax.invert_yaxis()
    # Mean
    m = np.concatenate([np.full(100, 18.5), np.full(150, 18.0), np.full(200, 18.2), np.full(50, 18.1)])
    y = m + np.random.normal(0, 0.15, 500)
    axes[0].plot(x, y, color=COLORS["data"], alpha=0.7)
    axes[0].plot(x, m, color=COLORS["seg"], lw=1.2)
    axes[0].set(xlabel="Time", ylabel="Magnitude", title="Change in mean")
    # Variance
    y = np.concatenate([
        np.random.normal(18.3, 0.05, 100), np.random.normal(18.3, 0.25, 150),
        np.random.normal(18.3, 0.1, 200), np.random.normal(18.3, 0.3, 50),
    ])
    axes[1].plot(x, y, color=COLORS["data"], alpha=0.7)
    axes[1].axhline(18.3, color=COLORS["seg"], ls="--", lw=1.2)
    axes[1].set(xlabel="Time", ylabel="Magnitude", title="Change in variance")
    # Mean & variance
    y = np.concatenate([
        np.random.normal(18.5, 0.08, 100), np.random.normal(18.0, 0.2, 150),
        np.random.normal(18.2, 0.12, 200), np.random.normal(18.1, 0.15, 50),
    ])
    m = np.concatenate([np.full(100, 18.5), np.full(150, 18.0), np.full(200, 18.2), np.full(50, 18.1)])
    axes[2].plot(x, y, color=COLORS["data"], alpha=0.7)
    axes[2].plot(x, m, color=COLORS["seg"], lw=1.2)
    axes[2].set(xlabel="Time", ylabel="Magnitude", title="Change in mean & variance")
    save("change-point-types.png")


# --- 2. AGN: PELT — change in mean (segment means only) ---
def _pelt_segments(signal, pen, model="normal", min_size=2):
    """Run PELT and return segment bounds (indices, last = n) and segment index per point."""
    n = len(signal)
    if n < 2 * min_size:
        return np.array([0, n]), np.zeros(n, dtype=int)
    sig = np.asarray(signal, dtype=float)
    if sig.ndim == 1:
        sig = sig.reshape(-1, 1)
    algo = rpt.Pelt(model=model, min_size=min_size).fit(sig)
    bkps = algo.predict(pen=pen)
    if bkps is None or len(bkps) == 0:
        bounds = np.array([0, n], dtype=int)
    else:
        # ruptures returns end indices of each regime (last element = n)
        bounds = np.unique(np.r_[0, np.asarray(bkps, dtype=int), n])
    seg_idx = np.repeat(np.arange(len(bounds) - 1), np.diff(bounds))
    return bounds, seg_idx


def plot_mean_segments(mjd, mag, magerr):
    """Plot light curve with PELT segmentation (mean): segment means (x-axis = MJD)."""
    n = len(mag)
    # Use lighter penalty so mean shifts are detected (PELT_PEN_MEAN < log(n) typically).
    pen = PELT_PEN_MEAN
    bounds, seg_idx = _pelt_segments(mag, pen, model="l2")
    # Changepoint MJD positions (bounds[1:-1]; bounds includes 0 and n)
    edges_mjd = mjd[bounds[1:-1]] if len(bounds) > 2 else np.array([])
    n_seg = len(bounds) - 1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.invert_yaxis()
    ax.scatter(mjd, mag, s=12, c=COLORS["data"], alpha=0.8)
    for ed in edges_mjd:
        ax.axvline(ed, color=COLORS["cpt"], lw=2, ls="--")
    for k in range(n_seg):
        mask = seg_idx == k
        if not np.any(mask):
            continue
        mjd_k = mjd[mask]
        mu = np.mean(mag[mask])
        ax.hlines(mu, mjd_k[0], mjd_k[-1], colors=COLORS["seg"], lw=2)
    ax.set(xlabel="MJD", ylabel="Magnitude (g)", title="PELT: change in mean (segment means)")
    ax.legend(["Data", "Changepoint", "Segment mean"], loc="upper right")
    ax.grid(True, alpha=0.3)
    save("change-in-mean.png")


# --- 3. AGN: PELT (variance-only: mean-subtracted) ---
def plot_pelt_variance(mjd, mag, magerr):
    """Plot light curve with PELT on centered data (variance): segment means ± 1σ."""
    n = len(mag)
    pen = np.log(n) if PELT_PEN is None else PELT_PEN
    centered = mag - np.mean(mag)
    bounds, seg_idx = _pelt_segments(centered, pen, model="normal")
    edges_mjd = mjd[bounds[1:-1]] if len(bounds) > 2 else np.array([])
    n_seg = len(bounds) - 1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.invert_yaxis()
    ax.scatter(mjd, mag, s=12, c=COLORS["data"], alpha=0.8)
    for ed in edges_mjd:
        ax.axvline(ed, color=COLORS["cpt"], lw=2, ls="--")
    for k in range(n_seg):
        mask = seg_idx == k
        if not np.any(mask):
            continue
        mjd_k = mjd[mask]
        mu, sig = np.mean(mag[mask]), np.std(mag[mask])
        ax.hlines(mu, mjd_k[0], mjd_k[-1], colors=COLORS["seg"], lw=2)
        ax.fill_between(mjd_k, mu - sig, mu + sig, color=COLORS["band"], alpha=0.25)
    ax.set(xlabel="MJD", ylabel="Magnitude (g)", title="PELT: change in variance (segment means ± 1σ)")
    ax.legend(["Data", "Changepoint", r"Mean ± 1$\sigma$"], loc="upper right")
    ax.grid(True, alpha=0.3)
    save("change-in-variance.png")


# --- 4. AGN: PELT (mean & variance) ---
def plot_pelt_meanvar(mjd, mag, magerr):
    """Plot light curve with PELT (mean & variance): segment means ± 1σ."""
    n = len(mag)
    pen = np.log(n) if PELT_PEN is None else PELT_PEN
    bounds, seg_idx = _pelt_segments(mag, pen, model="normal")
    edges_mjd = mjd[bounds[1:-1]] if len(bounds) > 2 else np.array([])
    n_seg = len(bounds) - 1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.invert_yaxis()
    ax.scatter(mjd, mag, s=12, c=COLORS["data"], alpha=0.8)
    for ed in edges_mjd:
        ax.axvline(ed, color=COLORS["cpt"], lw=2, ls="--")
    for k in range(n_seg):
        mask = seg_idx == k
        if not np.any(mask):
            continue
        mjd_k = mjd[mask]
        mu, sig = np.mean(mag[mask]), np.std(mag[mask])
        ax.hlines(mu, mjd_k[0], mjd_k[-1], colors=COLORS["seg"], lw=2)
        ax.fill_between(mjd_k, mu - sig, mu + sig, color=COLORS["band"], alpha=0.25)
    ax.set(xlabel="MJD", ylabel="Magnitude (g)", title="PELT: change in mean & variance (segment means ± 1σ)")
    ax.legend(["Data", "Changepoint", r"Mean ± 1$\sigma$"], loc="upper right")
    ax.grid(True, alpha=0.3)
    save("change-in-mean-variance.png")


# --- 5. Piecewise trend + harmonic (binary segmentation, adaptive-pooled strategy) ---
# Changepoints are found by binary segmentation: repeatedly apply AMOC (At Most One Change)
# on the current segments; at each step choose the split that most reduces pooled MSE
# (error proportional to points in each segment). Adaptive-pooled = criterion is
# (RSS_left + RSS_right)/n_seg within each segment, and (total RSS)/n globally.

# Note on Implementation:
# This code implements a classical Binary Segmentation approach for detecting
# multiple changepoints (Vostrikova, 1981; Bai, 1997; Bai & Perron, 1998, 2003).
#
# Each segment is modeled using a quadratic + harmonic regression.
# Changepoints are selected greedily by minimizing the pooled mean squared error
# (equivalently, the total residual sum of squares) across segments.
#
# Binary Segmentation is a well-established and widely used method in
# structural break and time series analysis. The harmonic regression basis
# and pooled-error criterion are used here for pedagogical purposes and for
# demonstration on irregularly sampled series such as AGN light curves.
#
# Binary Segmentation does not require regular sampling; it operates on ordered
# observations. Since the regression model is written explicitly as a function
# of time, irregular sampling is naturally accommodated.
#
# This is a straightforward and ilustrative implementation.


def _design(t_vals, p_index):
    """Design matrix: intercept, t, t², cos(2π t/p), sin(2π t/p)."""
    x = np.asarray(t_vals, dtype=float)
    return np.column_stack([
        np.ones_like(x), x, x**2,
        np.cos(2 * np.pi * x / p_index), np.sin(2 * np.pi * x / p_index),
    ])


def _rss_and_coeffs(t_vals, mag_vals, p_index):
    """Fit OLS and return RSS and coefficient vector."""
    X = _design(t_vals, p_index)
    c, _, _, _ = np.linalg.lstsq(X, mag_vals, rcond=None)
    rss = np.sum((mag_vals - X @ c) ** 2)
    return rss, c


def _amoc_one(t, mag, lo, hi, p_index, win):
    """
    Find best single changepoint in segment [lo, hi) by minimizing pooled MSE
    (ecm_c = (RSS1/n1 + RSS2/n2) weighted by points: (rmse1^2*t + rmse2^2*(n-t))/n).
    Same as minimizing (RSS_left + RSS_right) / n_seg for this segment.
    """
    n_seg = hi - lo
    if n_seg < 2 * win:
        return None, np.inf
    t_seg = t[lo:hi]
    mag_seg = mag[lo:hi]
    ecm_best = np.inf
    best_i = None
    for i in range(win, n_seg - win):
        n_left, n_right = i + 1, n_seg - (i + 1)
        rss_left, _ = _rss_and_coeffs(t_seg[: i + 1], mag_seg[: i + 1], p_index)
        rss_right, _ = _rss_and_coeffs(t_seg[i + 1 :], mag_seg[i + 1 :], p_index)
        rmse1_sq = rss_left / n_left
        rmse2_sq = rss_right / n_right
        ecm_c = (rmse1_sq * n_left + rmse2_sq * n_right) / n_seg  # pooled MSE
        if ecm_c < ecm_best:
            ecm_best = ecm_c
            best_i = i
    # Return first index of right segment (so bounds [lo, cpt, hi] give segments [lo:cpt], [cpt:hi])
    return (lo + best_i + 1) if best_i is not None else None, ecm_best


def _total_rss_with_splits(t, mag, p_index, changepoints):
    """Total RSS when fitting the harmonic model in each segment defined by changepoints."""
    bounds = [0] + sorted(changepoints) + [len(t)]
    total = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 3:
            return np.inf
        rss, _ = _rss_and_coeffs(t[a:b], mag[a:b], p_index)
        total += rss
    return total


def _pooled_mse_with_splits(t, mag, p_index, changepoints):
    """Pooled MSE = (sum of segment RSS) / n, proportional to points in each segment (like R ecm_c)."""
    n = len(t)
    if n == 0:
        return np.inf
    return _total_rss_with_splits(t, mag, p_index, changepoints) / n


def piecewise_changepoints_binary(t, mag, p_index, min_frac=0.10, max_cpts=5):
    """
    Binary segmentation: find changepoints by repeated AMOC.
    Like R's AMOC loop, but after finding one change we search again in each segment.
    Returns sorted list of changepoint indices (between segments).
    """
    n = len(t)
    win = max(2, int(min_frac * n))
    changepoints = []
    segments = [(0, n)]

    for _ in range(max_cpts):
        best_new_cpt = None
        best_ecm = np.inf
        best_seg_idx = None

        for seg_idx, (a, b) in enumerate(segments):
            cpt, _ = _amoc_one(t, mag, a, b, p_index, win)
            if cpt is None:
                continue
            # Pooled MSE if we add this changepoint (proportional to points in each segment)
            new_cpts = sorted(changepoints + [cpt])
            ecm = _pooled_mse_with_splits(t, mag, p_index, new_cpts)
            if ecm < best_ecm:
                best_ecm = ecm
                best_new_cpt = cpt
                best_seg_idx = seg_idx

        if best_new_cpt is None:
            break
        ecm_without = _pooled_mse_with_splits(t, mag, p_index, changepoints)
        if best_ecm >= ecm_without:
            break
        changepoints = sorted(changepoints + [best_new_cpt])
        a, b = segments[best_seg_idx]
        segments.pop(best_seg_idx)
        if best_new_cpt - a >= 2 * win:
            segments.append((a, best_new_cpt))
        if b - best_new_cpt >= 2 * win:
            segments.append((best_new_cpt, b))
        segments.sort(key=lambda s: s[0])

    return sorted(changepoints)


def plot_piecewise(t, mag, mjd, n):
    """Piecewise trend + 1 harmonic; changepoints found by binary segmentation (AMOC-style)."""
    # Harmonic period in time-index units (period_days / mean_sampling_interval_days)
    p_index = HARMONIC_PERIOD_DAYS / np.mean(np.diff(mjd))
    win = max(2, int(PIECEWISE_MIN_FRAC * n))
    cpts = piecewise_changepoints_binary(
        t, mag, p_index, min_frac=PIECEWISE_MIN_FRAC, max_cpts=PIECEWISE_MAX_CPTS
    )
    bounds = [0] + cpts + [n]
    n_seg = len(bounds) - 1

    coeffs_list = []
    fits_list = []
    t_list = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        t_seg = t[a:b]
        mag_seg = mag[a:b]
        _, c = _rss_and_coeffs(t_seg, mag_seg, p_index)
        coeffs_list.append(c)
        X = _design(t_seg, p_index)
        fits_list.append(X @ c)
        t_list.append(t_seg)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes
    ax1.invert_yaxis()
    ax1.scatter(t, mag, s=8, c=COLORS["data"], alpha=0.6)
    seg_colors = [COLORS["data"], COLORS["seg"], "#8B0000", "#2E86AB", "#6A994E", "#E07A5F", "#3D405B", "#81B29A"]
    seg_colors = (seg_colors * ((n_seg // len(seg_colors)) + 1))[:n_seg]
    for k, (t_seg, fit_vals) in enumerate(zip(t_list, fits_list)):
        ax1.plot(t_seg, fit_vals, color=seg_colors[k], lw=1.2, label=f"Seg {k + 1}")
    for cpt in cpts:
        ax1.axvline(t[cpt], color=COLORS["cpt"], ls="--")
    ax1.set(xlabel="Time index", ylabel="Magnitude (g)", title="Piecewise: trend + 1 harmonic (binary seg)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    terms = ["t", "t²", "cos", "sin"]
    n_terms = len(terms)
    x = np.arange(n_terms)
    w = 0.8 / max(n_seg, 1)
    for k, c in enumerate(coeffs_list):
        off = (k - (n_seg - 1) / 2) * w
        ax2.bar(x + off, c[1:], w, label=f"Seg {k + 1}", color=seg_colors[k], alpha=0.7)
    ax2.axhline(0, color="k", ls="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(terms)
    ax2.set(ylabel="Coefficient", title="Coefficients by segment")
    ax2.legend()
    save("piecewise-models.png")


def main():
    print("AGN teaching example: ZTF19acnskyy")
    print("Fetching g-band light curve from IRSA...")
    ts, t, mag, mjd, magerr = load_agn_lightcurve()
    n = len(t)
    print(f"  {n} points")

    print("Plotting...")
    plot_change_types()
    plot_mean_segments(mjd, mag, magerr)
    plot_pelt_variance(mjd, mag, magerr)
    plot_pelt_meanvar(mjd, mag, magerr)
    plot_piecewise(t, mag, mjd, n)
    print("Done. Figures in", OUT)


if __name__ == "__main__":
    main()
