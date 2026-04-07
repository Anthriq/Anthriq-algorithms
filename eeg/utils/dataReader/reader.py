"""
reader.py — EEG data loading with automatic format detection and metadata inference.

Supports: EDF/BDF, CSV/TSV, MAT, BrainVision (.vhdr), EEGLab (.set), FIF
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import mne
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".edf": "edf",
    ".bdf": "bdf",
    ".fif": "fif",
    ".vhdr": "brainvision",
    ".set": "eeglab",
    ".mat": "mat",
    ".csv": "csv",
    ".tsv": "tsv",
}

# Typical EEG amplitude range in µV (median scalp EEG is ~10–100 µV)
_UV_LOWER, _UV_UPPER = 1e-7, 1e-3   # in Volts: 0.1 µV – 1 mV
_MV_LOWER, _MV_UPPER = 1e-4, 1e-1   # in Volts: 0.1 mV – 100 mV

# README-like filenames to scan for metadata
_README_NAMES = [
    "README.md", "README.txt", "README", "readme.md", "readme.txt",
    "dataset_description.json",  # BIDS
    "participants.tsv",           # BIDS
]

_TASK_KEYWORDS    = ["task", "erp", "stimulus", "event", "response", "oddball",
                     "p300", "ssvep", "motor", "imagin"]
_REST_KEYWORDS    = ["rest", "resting", "baseline", "eyes open", "eyes closed"]
_EPOCH_KEYWORDS   = ["epoch", "epoched", "segmented", "trial"]
_CONTINUOUS_KEYWORDS = ["continuous", "raw", "streaming"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(path: str | Path) -> tuple[mne.io.BaseRaw | mne.BaseEpochs, dict[str, Any]]:
    """
    Load an EEG file and infer metadata.

    Parameters
    ----------
    path : str or Path
        Path to the EEG file.

    Returns
    -------
    data : mne.io.BaseRaw or mne.BaseEpochs
        Loaded MNE object.
    metadata : dict
        Inferred metadata including format, units, task type, etc.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    fmt = _detect_format(path)
    data = _load_by_format(path, fmt)
    metadata = _build_metadata(path, fmt, data)

    _print_summary(path, fmt, metadata)
    return data, metadata


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension: '{ext}'\n"
            f"Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
        )
    return SUPPORTED_EXTENSIONS[ext]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_by_format(path: Path, fmt: str) -> mne.io.BaseRaw | mne.BaseEpochs:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if fmt == "edf":
            return mne.io.read_raw_edf(str(path), preload=True, verbose=False)

        if fmt == "bdf":
            return mne.io.read_raw_bdf(str(path), preload=True, verbose=False)

        if fmt == "fif":
            try:
                return mne.io.read_raw_fif(str(path), preload=True, verbose=False)
            except Exception:
                return mne.read_epochs(str(path), verbose=False)

        if fmt == "brainvision":
            return mne.io.read_raw_brainvision(str(path), preload=True, verbose=False)

        if fmt == "eeglab":
            try:
                return mne.io.read_raw_eeglab(str(path), preload=True, verbose=False)
            except Exception:
                return mne.read_epochs_eeglab(str(path), verbose=False)

        if fmt == "mat":
            return _load_mat(path)

        if fmt in ("csv", "tsv"):
            return _load_delimited(path, fmt)

    raise ValueError(f"No loader available for format: {fmt}")


def _load_mat(path: Path) -> mne.io.BaseRaw:
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("scipy is required to load .mat files: pip install scipy")

    mat = sio.loadmat(str(path), squeeze_me=True)

    # Try common variable names used in EEG MATLAB files
    data_keys = [k for k in mat if not k.startswith("_")]
    eeg_key = next(
        (k for k in data_keys if k.lower() in ("eeg", "data", "signal", "x")),
        data_keys[0] if data_keys else None,
    )
    if eeg_key is None:
        raise ValueError(f"Cannot find EEG data array in .mat file. Keys: {list(mat.keys())}")

    arr = np.atleast_2d(mat[eeg_key])
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T  # ensure (channels, samples)

    sfreq = float(mat.get("srate", mat.get("fs", mat.get("Fs", mat.get("sfreq", 256)))))
    ch_names = [f"CH{i+1}" for i in range(arr.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(arr, info, verbose=False)


def _load_delimited(path: Path, fmt: str) -> mne.io.BaseRaw:
    import pandas as pd

    sep = "," if fmt == "csv" else "\t"
    df = pd.read_csv(str(path), sep=sep)

    # Heuristic: drop non-numeric or time/marker columns
    non_data_patterns = re.compile(r"time|timestamp|marker|event|label|trigger|stim",
                                   re.IGNORECASE)
    ch_cols = [c for c in df.columns if not non_data_patterns.search(str(c))
               and pd.api.types.is_numeric_dtype(df[c])]

    if not ch_cols:
        raise ValueError("Could not identify EEG channel columns in the CSV/TSV file.")

    arr = df[ch_cols].values.T.astype(float)

    # Try to infer sample rate from a time column
    sfreq = 256.0
    time_col = next((c for c in df.columns if re.search(r"^time", str(c), re.IGNORECASE)), None)
    if time_col is not None:
        diffs = np.diff(df[time_col].values)
        median_dt = float(np.median(diffs[diffs > 0]))
        if median_dt > 0:
            sfreq = round(1.0 / median_dt)

    info = mne.create_info(ch_names=ch_cols, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(arr, info, verbose=False)

    # Attach marker channel as annotations if present
    marker_col = next(
        (c for c in df.columns if non_data_patterns.search(str(c))
         and pd.api.types.is_numeric_dtype(df[c])),
        None,
    )
    if marker_col is not None:
        events_mask = df[marker_col] != 0
        if events_mask.any():
            times = np.arange(len(df)) / sfreq
            onsets = times[events_mask.values]
            descriptions = df.loc[events_mask, marker_col].astype(str).values
            raw.set_annotations(mne.Annotations(onset=onsets, duration=0.0,
                                                description=descriptions))
    return raw


# ---------------------------------------------------------------------------
# Metadata inference
# ---------------------------------------------------------------------------

def _build_metadata(path: Path, fmt: str, data: mne.io.BaseRaw | mne.BaseEpochs) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "file": str(path),
        "format": fmt,
        "n_channels": len(data.ch_names),
        "channel_names": data.ch_names,
        "sfreq": data.info["sfreq"],
        "duration_s": None,
        "unit": None,
        "unit_confidence": None,
        "data_type": None,         # "continuous" | "epoched"
        "task_type": None,         # "task" | "rest" | "unknown"
        "has_markers": False,
        "marker_description": None,
        "n_subjects": None,
        "n_sessions": None,
    }

    # Data type
    if isinstance(data, mne.BaseEpochs):
        meta["data_type"] = "epoched"
        meta["duration_s"] = (data.times[-1] - data.times[0]) * len(data)
    else:
        meta["data_type"] = "continuous"
        meta["duration_s"] = data.n_times / data.info["sfreq"]

    # Markers / annotations
    if isinstance(data, mne.BaseEpochs):
        meta["has_markers"] = True
        meta["marker_description"] = f"{len(data)} epochs"
    elif isinstance(data, mne.io.BaseRaw):
        if len(data.annotations) > 0:
            meta["has_markers"] = True
            unique_annots = list(set(data.annotations.description))
            meta["marker_description"] = f"{len(data.annotations)} annotations, types: {unique_annots}"
        else:
            try:
                events, event_id = mne.events_from_annotations(data, verbose=False)
                if len(events) > 0:
                    meta["has_markers"] = True
                    meta["marker_description"] = f"{len(events)} events, ids: {list(event_id.keys())}"
            except Exception:
                pass

    # Unit inference from median signal amplitude
    try:
        if isinstance(data, mne.BaseEpochs):
            sample = data.get_data()[:5].ravel()
        else:
            n = min(data.n_times, int(data.info["sfreq"] * 10))
            sample = data.get_data()[:, :n].ravel()

        median_abs = float(np.median(np.abs(sample[sample != 0])))

        if _UV_LOWER <= median_abs <= _UV_UPPER:
            meta["unit"] = "Volts (likely µV-scale signal stored in V)"
            meta["unit_confidence"] = "high"
        elif 1e-4 <= median_abs <= 5e-4:
            meta["unit"] = "Likely µV (values already in µV)"
            meta["unit_confidence"] = "medium"
        elif median_abs > 5e-4:
            meta["unit"] = "Possibly mV or raw ADC counts"
            meta["unit_confidence"] = "low"
        else:
            meta["unit"] = "Very small — possibly already in V (standard MNE)"
            meta["unit_confidence"] = "medium"
    except Exception:
        meta["unit"] = "Could not infer"
        meta["unit_confidence"] = "none"

    # Scan README / sidecar files
    readme_meta = _scan_readme_files(path.parent)
    meta.update({k: v for k, v in readme_meta.items() if v is not None})

    # Fallback: infer task type from filename if not found in README
    if meta["task_type"] is None:
        fname_lower = path.stem.lower()
        if any(kw in fname_lower for kw in _TASK_KEYWORDS):
            meta["task_type"] = "task"
        elif any(kw in fname_lower for kw in _REST_KEYWORDS):
            meta["task_type"] = "rest"
        else:
            meta["task_type"] = "unknown"

    return meta


def _scan_readme_files(directory: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_type": None,
        "n_subjects": None,
        "n_sessions": None,
    }

    for name in _README_NAMES:
        candidate = directory / name
        if not candidate.exists():
            # Search one level up too (e.g. subject sub-folder)
            candidate = directory.parent / name
        if not candidate.exists():
            continue

        if candidate.suffix == ".json":
            result.update(_parse_bids_json(candidate))
        elif candidate.suffix == ".tsv" and "participants" in candidate.name:
            result.update(_parse_participants_tsv(candidate))
        else:
            result.update(_parse_readme_text(candidate))

    return result


def _parse_readme_text(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"task_type": None, "n_subjects": None, "n_sessions": None}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return result

    if any(kw in text for kw in _TASK_KEYWORDS):
        result["task_type"] = "task"
    elif any(kw in text for kw in _REST_KEYWORDS):
        result["task_type"] = "rest"

    subj_match = re.search(r"(\d+)\s*(?:subject|participant|person)", text)
    if subj_match:
        result["n_subjects"] = int(subj_match.group(1))

    sess_match = re.search(r"(\d+)\s*session", text)
    if sess_match:
        result["n_sessions"] = int(sess_match.group(1))

    return result


def _parse_bids_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"task_type": None, "n_subjects": None, "n_sessions": None}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        task_name = str(obj.get("TaskName", "")).lower()
        if any(kw in task_name for kw in _REST_KEYWORDS):
            result["task_type"] = "rest"
        elif task_name:
            result["task_type"] = "task"
    except Exception:
        pass
    return result


def _parse_participants_tsv(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"task_type": None, "n_subjects": None, "n_sessions": None}
    try:
        import pandas as pd
        df = pd.read_csv(str(path), sep="\t")
        result["n_subjects"] = len(df)
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(path: Path, fmt: str, meta: dict[str, Any]) -> None:
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  anthriq-eeg | Data Reader Summary")
    print(sep)
    print(f"  File       : {path.name}")
    print(f"  Format     : {fmt.upper()}")
    print(f"  Type       : {meta['data_type']}")
    print(f"  Channels   : {meta['n_channels']}")
    print(f"  Sfreq      : {meta['sfreq']} Hz")
    if meta["duration_s"] is not None:
        print(f"  Duration   : {meta['duration_s']:.1f} s")
    print(f"  Task type  : {meta['task_type']}")
    print(f"  Markers    : {'Yes — ' + meta['marker_description'] if meta['has_markers'] else 'None detected'}")
    print(f"  Subjects   : {meta['n_subjects'] or 'not found in README'}")
    print(f"  Sessions   : {meta['n_sessions'] or 'not found in README'}")
    print(sep)
    print(f"  Unit (inferred): {meta['unit']}")
    print(f"  Confidence      : {meta['unit_confidence']}")
    print(f"\n  Please verify the unit manually before proceeding.")
    print(sep + "\n")
