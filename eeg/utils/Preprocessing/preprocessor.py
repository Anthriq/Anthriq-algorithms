"""
preprocessor.py — EEG preprocessing: bandpass filter, notch filter, and event-based epoching.
"""

from __future__ import annotations

from typing import Any

import mne


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    metadata: dict[str, Any] | None = None,
    *,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    line_freq: float | None = None,
    epoch_tmin: float = -0.2,
    epoch_tmax: float = 0.8,
    baseline: tuple[float | None, float] = (None, 0.0),
) -> mne.io.BaseRaw | mne.BaseEpochs:
    """
    Apply standard EEG preprocessing.

    Steps:
      1. Bandpass filter (default 0.5–40 Hz)
      2. Notch filter at power line frequency (prompted from user)
      3. If continuous data with markers: epoch around events

    Parameters
    ----------
    data : mne.io.BaseRaw or mne.BaseEpochs
        Loaded EEG data.
    metadata : dict, optional
        Metadata dict returned by ``reader.load()``. Used to determine
        whether markers exist and whether the data is already epoched.
    l_freq : float
        Bandpass lower cutoff in Hz (default 0.5).
    h_freq : float
        Bandpass upper cutoff in Hz (default 40.0).
    line_freq : float or None
        Power line frequency in Hz (50 or 60). If None, the user is prompted.
    epoch_tmin : float
        Epoch start relative to event onset in seconds (default -0.2).
    epoch_tmax : float
        Epoch end relative to event onset in seconds (default 0.8).
    baseline : tuple
        Baseline correction window (default (None, 0) = from start to event onset).

    Returns
    -------
    mne.io.BaseRaw or mne.BaseEpochs
        Preprocessed data.
    """
    metadata = metadata or {}

    line_freq = _resolve_line_freq(line_freq)

    print(f"[preprocess] Bandpass filter: {l_freq}–{h_freq} Hz")
    data = _apply_bandpass(data, l_freq, h_freq)

    print(f"[preprocess] Notch filter: {line_freq} Hz")
    data = _apply_notch(data, line_freq)

    # Epoch continuous data if markers are present and data is not already epoched
    if isinstance(data, mne.io.BaseRaw) and metadata.get("has_markers", False):
        print(f"[preprocess] Markers detected — epoching {epoch_tmin}s to {epoch_tmax}s around events")
        data = _epoch(data, epoch_tmin, epoch_tmax, baseline)

    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_line_freq(line_freq: float | None) -> float:
    if line_freq is not None:
        return float(line_freq)

    print("\nPower line frequency not set.")
    print("  [1] 50 Hz  (Europe, Asia, Africa, most of the world)")
    print("  [2] 60 Hz  (North America, parts of South America, Japan)")
    while True:
        choice = input("Enter choice [1/2] or type the frequency directly: ").strip()
        if choice == "1":
            return 50.0
        if choice == "2":
            return 60.0
        try:
            val = float(choice)
            if 40.0 < val < 70.0:
                return val
        except ValueError:
            pass
        print("  Invalid input. Please enter 1, 2, or a frequency value (e.g. 50 or 60).")


def _apply_bandpass(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    l_freq: float,
    h_freq: float,
) -> mne.io.BaseRaw | mne.BaseEpochs:
    return data.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_window="hamming",
        verbose=False,
    )


def _apply_notch(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    line_freq: float,
) -> mne.io.BaseRaw | mne.BaseEpochs:
    sfreq = data.info["sfreq"]
    # Include harmonics up to Nyquist
    freqs = [line_freq * i for i in range(1, 10) if line_freq * i < sfreq / 2]
    return data.notch_filter(freqs=freqs, verbose=False)


def _epoch(
    raw: mne.io.BaseRaw,
    tmin: float,
    tmax: float,
    baseline: tuple[float | None, float],
) -> mne.BaseEpochs:
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
    except Exception as exc:
        print(f"[preprocess] Warning: could not extract events — {exc}. Returning raw.")
        return raw  # type: ignore[return-value]

    if len(events) == 0:
        print("[preprocess] Warning: no events found in annotations. Returning raw.")
        return raw  # type: ignore[return-value]

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
    )
    print(f"[preprocess] Created {len(epochs)} epochs across {len(event_id)} event type(s).")
    return epochs
