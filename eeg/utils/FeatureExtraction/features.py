"""
features.py — EEG feature extraction.

Produces a pandas DataFrame with rows = (epoch × channel) and columns = features.

Time-domain
-----------
  hjorth_activity, hjorth_mobility, hjorth_complexity
  rms, mav, waveform_length, zero_crossings, variance,
  skewness, kurtosis, peak_to_peak, mean_absolute_deviation

Frequency-domain
----------------
  total_power, median_frequency, mean_frequency,
  band_power_delta, band_power_theta, band_power_alpha,
  band_power_beta, band_power_gamma, spectral_entropy

Nonlinear
---------
  sample_entropy, approximate_entropy
"""

from __future__ import annotations

import warnings
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy import signal, stats


# ---------------------------------------------------------------------------
# Band definitions (Hz)
# ---------------------------------------------------------------------------

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    *,
    sfreq: float | None = None,
) -> pd.DataFrame:
    """
    Extract all features from EEG data.

    Parameters
    ----------
    data : mne.io.BaseRaw or mne.BaseEpochs
        Preprocessed EEG data.
    sfreq : float, optional
        Sampling frequency. Inferred from ``data.info`` if not provided.

    Returns
    -------
    pd.DataFrame
        Rows = (epoch, channel), columns = feature names.
        For continuous Raw data a single pseudo-epoch is used.
    """
    fs = sfreq or data.info["sfreq"]
    ch_names = data.ch_names

    if isinstance(data, mne.BaseEpochs):
        # shape: (n_epochs, n_channels, n_times)
        arrays = data.get_data()
        epoch_labels = [f"epoch_{i}" for i in range(arrays.shape[0])]
    else:
        # Treat entire recording as one epoch
        arrays = data.get_data()[np.newaxis, :, :]  # (1, n_ch, n_times)
        epoch_labels = ["continuous"]

    records = []
    for ep_idx, epoch_arr in enumerate(arrays):
        for ch_idx, ch_signal in enumerate(epoch_arr):
            row = _compute_all(ch_signal, fs)
            row["epoch"] = epoch_labels[ep_idx]
            row["channel"] = ch_names[ch_idx]
            records.append(row)

    df = pd.DataFrame(records)
    # Reorder: epoch, channel first, then features alphabetically
    id_cols = ["epoch", "channel"]
    feat_cols = sorted(c for c in df.columns if c not in id_cols)
    return df[id_cols + feat_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature dispatcher
# ---------------------------------------------------------------------------

def _compute_all(x: np.ndarray, fs: float) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(_time_domain(x))
    feats.update(_frequency_domain(x, fs))
    feats.update(_nonlinear(x))
    return feats


# ---------------------------------------------------------------------------
# Time-domain features
# ---------------------------------------------------------------------------

def _time_domain(x: np.ndarray) -> dict[str, float]:
    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x   = float(np.var(x, ddof=1))
    var_dx  = float(np.var(dx, ddof=1))
    var_ddx = float(np.var(ddx, ddof=1))

    # Hjorth parameters
    activity   = var_x
    mobility   = float(np.sqrt(var_dx / var_x)) if var_x > 0 else 0.0
    mob_dx     = float(np.sqrt(var_ddx / var_dx)) if var_dx > 0 else 0.0
    complexity = (mob_dx / mobility) if mobility > 0 else 0.0

    return {
        "hjorth_activity":        activity,
        "hjorth_mobility":        mobility,
        "hjorth_complexity":      complexity,
        "rms":                    float(np.sqrt(np.mean(x ** 2))),
        "mav":                    float(np.mean(np.abs(x))),
        "waveform_length":        float(np.sum(np.abs(dx))),
        "zero_crossings":         int(np.sum(np.diff(np.sign(x)) != 0)),
        "variance":               var_x,
        "skewness":               float(stats.skew(x)),
        "kurtosis":               float(stats.kurtosis(x)),
        "peak_to_peak":           float(np.ptp(x)),
        "mean_absolute_deviation": float(np.mean(np.abs(x - np.mean(x)))),
    }


# ---------------------------------------------------------------------------
# Frequency-domain features
# ---------------------------------------------------------------------------

def _frequency_domain(x: np.ndarray, fs: float) -> dict[str, float]:
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(len(x), 256))

    total_power    = float(np.trapezoid(psd, freqs))
    cumulative     = np.cumsum(psd)
    median_freq    = float(freqs[np.searchsorted(cumulative, cumulative[-1] / 2)])
    mean_freq      = float(np.sum(freqs * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0.0

    feats: dict[str, float] = {
        "total_power":    total_power,
        "median_frequency": median_freq,
        "mean_frequency":   mean_freq,
    }

    # Band powers
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        feats[f"band_power_{band}"] = float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0

    # Spectral entropy (normalised Shannon entropy of the PSD)
    psd_norm = psd / (np.sum(psd) + 1e-12)
    spectral_ent = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
    feats["spectral_entropy"] = spectral_ent

    return feats


# ---------------------------------------------------------------------------
# Nonlinear features
# ---------------------------------------------------------------------------

def _nonlinear(x: np.ndarray) -> dict[str, float]:
    return {
        "sample_entropy":      _sample_entropy(x),
        "approximate_entropy": _approximate_entropy(x),
    }


def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Sample Entropy (SampEn).

    Parameters
    ----------
    m : int
        Template length (default 2).
    r_factor : float
        Tolerance as fraction of signal std (default 0.2).
    """
    N = len(x)
    r = r_factor * float(np.std(x, ddof=1))
    if r == 0 or N < m + 2:
        return 0.0

    def _count_matches(template_len: int) -> int:
        count = 0
        for i in range(N - template_len):
            template = x[i: i + template_len]
            for j in range(i + 1, N - template_len):
                if np.max(np.abs(x[j: j + template_len] - template)) < r:
                    count += 1
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)

    if B == 0:
        return 0.0
    return float(-np.log(A / B))


def _approximate_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Approximate Entropy (ApEn).

    Parameters
    ----------
    m : int
        Template length (default 2).
    r_factor : float
        Tolerance as fraction of signal std (default 0.2).
    """
    N = len(x)
    r = r_factor * float(np.std(x, ddof=1))
    if r == 0 or N < m + 1:
        return 0.0

    def _phi(template_len: int) -> float:
        count = np.zeros(N - template_len + 1)
        for i in range(N - template_len + 1):
            template = x[i: i + template_len]
            for j in range(N - template_len + 1):
                if np.max(np.abs(x[j: j + template_len] - template)) <= r:
                    count[i] += 1
        count = count / (N - template_len + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.mean(np.log(count + 1e-12)))

    return _phi(m) - _phi(m + 1)
