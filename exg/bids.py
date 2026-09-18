# Copyright 2026 Nexstem India Private Limited (trading as Anthriq)
# Licensed under the Apache License, Version 2.0. See the LICENSE file.

"""
Reading BIDS datasets, and the BrainVision files inside them.

BIDS -- the Brain Imaging Data Structure -- is a convention for laying out
neuroscience data so that a dataset is self-describing: the folder names carry
the subject, session and task, and small sidecar files say what the channels
are, what units they are in, and when each event happened. It is worth knowing
because a great deal of public EEG data is published this way, so a script that
reads BIDS can be pointed at datasets nobody here recorded.

A minimal EEG dataset looks like this::

    my_dataset/
    |-- dataset_description.json     what this dataset is
    |-- participants.tsv             one row per subject
    `-- sub-01/
        `-- eeg/
            |-- sub-01_task-alpha_eeg.vhdr      header: channels, rate, file layout
            |-- sub-01_task-alpha_eeg.eeg       the samples themselves, binary
            |-- sub-01_task-alpha_eeg.vmrk      markers, as written by the recorder
            |-- sub-01_task-alpha_eeg.json      sampling rate, reference, filters
            |-- sub-01_task-alpha_channels.tsv  name, type and units per channel
            `-- sub-01_task-alpha_events.tsv    onset and duration of each event

BIDS allows three formats for the signal itself: BrainVision, EDF and EEGLAB.
This module reads **BrainVision**, which is the one most BIDS EEG datasets use
and by far the easiest to read honestly -- the header is an INI file, the data
is a flat binary array, and the markers are a text list.

Why not use a library
---------------------
MNE-Python reads all of this in one line, and if you are building something real
you should use it. These scripts exist so you can see what the analysis is
doing, and a reader you can follow start to finish is part of that. The format
is genuinely simple; there is nothing here you could not work out from the
files themselves in an afternoon.

References
----------
Gorgolewski, K. J., et al. (2016). The brain imaging data structure, a format
    for organizing and describing outputs of neuroimaging experiments.
    Scientific Data, 3, 160044.
Pernet, C. R., et al. (2019). EEG-BIDS, an extension to the brain imaging data
    structure for electroencephalography. Scientific Data, 6, 103.
"""

from __future__ import annotations

import configparser
import csv
import json
import warnings
from pathlib import Path

import numpy as np

from .io import Marker, Recording

__all__ = [
    "read_brainvision",
    "read_bids_recording",
    "find_bids_recordings",
    "describe_bids_dataset",
]

# BrainVision stores samples as one of these. The names are the ones the header
# uses; the values are the matching NumPy types.
_BINARY_FORMATS = {
    "int_16": np.dtype("<i2"),
    "int_32": np.dtype("<i4"),
    "ieee_float_32": np.dtype("<f4"),
    "ieee_float_64": np.dtype("<f8"),
}

# Channel units as they appear in a BrainVision header or a channels.tsv,
# mapped to the multiplier that takes them to microvolts.
_UNIT_TO_MICROVOLTS = {
    "v": 1e6, "volt": 1e6, "volts": 1e6,
    "mv": 1e3, "millivolt": 1e3,
    "uv": 1.0, "µv": 1.0, "μv": 1.0, "microvolt": 1.0,
    "nv": 1e-3,
}


def _read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated file with a header row into a list of dicts.

    BIDS uses TSV rather than CSV throughout, and writes a literal ``n/a`` for
    a missing value.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _as_float(text: str | None, default: float = float("nan")) -> float:
    """Parse a TSV field, treating BIDS's ``n/a`` as missing."""
    if text is None or text.strip().lower() in ("", "n/a", "na", "nan"):
        return default
    try:
        return float(text)
    except ValueError:
        return default


# --------------------------------------------------------------------------
# BrainVision
# --------------------------------------------------------------------------

def read_brainvision(vhdr_path: str | Path) -> tuple[np.ndarray, list[str], float, list[Marker]]:
    """Read a BrainVision recording.

    Three files work together, all sharing a stem:

    * ``.vhdr`` -- the header. An INI file naming the other two, giving the
      channel count, the sampling interval, the binary layout, and one line per
      channel with its name, its resolution and its units.
    * ``.eeg`` -- the samples, as a flat binary array with no header of its own.
      Multiplexed, meaning the channels are interleaved sample by sample:
      ``ch1[0], ch2[0], ch3[0], ch1[1], ch2[1], ...``
    * ``.vmrk`` -- markers, as a text list. Often empty in a BIDS dataset,
      where the events live in ``events.tsv`` instead.

    The header's per-channel **resolution** is the number the stored integers
    must be multiplied by to reach the stated unit, which is how a 16-bit file
    represents microvolt-scale signals without losing precision.

    Parameters
    ----------
    vhdr_path : path
        The ``.vhdr`` header file.

    Returns
    -------
    data : ndarray
        Shape (n_channels, n_samples), in **microvolts**.
    channels : list of str
        Channel names, in file order.
    fs : float
        Sampling rate in Hz.
    markers : list of Marker
        From the ``.vmrk`` file. Empty if it has none or is absent.
    """
    vhdr_path = Path(vhdr_path).expanduser().resolve()
    if not vhdr_path.is_file():
        raise FileNotFoundError(vhdr_path)

    # The header is INI-shaped, but with two quirks.
    #
    # First, it opens with a magic line -- "Brain Vision Data Exchange Header
    # File Version 1.0" -- before any [Section], which an INI parser rejects
    # outright. Every real header has it, so skip everything before the first
    # bracketed section rather than trying to parse it.
    #
    # Second, its channel lines are "Ch1=name,ref,resolution,unit", so the
    # value is a comma-separated list we split ourselves. Interpolation is off
    # because a channel name or a comment containing '%' would otherwise raise.
    text = vhdr_path.read_text(encoding="utf-8", errors="replace")
    first_section = text.find("[")
    if first_section < 0:
        raise ValueError(
            f"{vhdr_path.name} contains no [Section] headers, so it is not a "
            "BrainVision header file."
        )

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str  # keep Ch1/Ch2 case as written
    parser.read_string(text[first_section:], source=str(vhdr_path))

    common = parser["Common Infos"]
    n_channels = int(common["NumberOfChannels"])

    # The header gives the interval between samples in microseconds, so the
    # rate is its reciprocal. Stated this way round because the interval is
    # what the recorder actually knows.
    sampling_interval_us = float(common["SamplingInterval"])
    fs = 1e6 / sampling_interval_us

    data_file = vhdr_path.parent / common["DataFile"]
    if not data_file.is_file():
        raise FileNotFoundError(
            f"{vhdr_path.name} names its data file as {common['DataFile']!r}, "
            f"which is not in {vhdr_path.parent}. A BrainVision recording is "
            "three files that travel together."
        )

    orientation = common.get("DataOrientation", "MULTIPLEXED").upper()
    data_format = common.get("DataFormat", "BINARY").upper()
    if data_format != "BINARY":
        raise NotImplementedError(
            f"{vhdr_path.name} is in {data_format} format. Only BINARY is "
            "supported here; ASCII BrainVision files are rare and can be "
            "converted with any of the usual tools."
        )

    binary = parser["Binary Infos"]["BinaryFormat"].strip().lower()
    if binary not in _BINARY_FORMATS:
        raise NotImplementedError(
            f"Unsupported BrainVision binary format {binary!r}. "
            f"Known: {sorted(_BINARY_FORMATS)}"
        )
    dtype = _BINARY_FORMATS[binary]

    # Per-channel name, resolution and unit, from lines like
    #   Ch1=O1,,0.1,µV
    # The second field is a per-channel reference, usually blank.
    names: list[str] = []
    resolutions: list[float] = []
    units: list[str] = []
    channel_section = parser["Channel Infos"] if parser.has_section("Channel Infos") else {}
    for index in range(1, n_channels + 1):
        raw = channel_section.get(f"Ch{index}", "")
        fields = [f.strip() for f in raw.split(",")]
        names.append(fields[0] if fields and fields[0] else f"ch{index}")
        resolutions.append(float(fields[2]) if len(fields) > 2 and fields[2] else 1.0)
        units.append(fields[3] if len(fields) > 3 and fields[3] else "uV")

    raw = np.fromfile(data_file, dtype=dtype)
    if raw.size % n_channels:
        warnings.warn(
            f"{data_file.name} holds {raw.size} values, which is not a whole "
            f"number of samples across {n_channels} channels. Truncating the "
            "incomplete final sample.",
            stacklevel=2,
        )
        raw = raw[: raw.size - (raw.size % n_channels)]

    if orientation == "MULTIPLEXED":
        # Interleaved: reshape to (n_samples, n_channels) then transpose.
        data = raw.reshape(-1, n_channels).T.astype(float)
    elif orientation == "VECTORIZED":
        # Channel by channel: each channel's whole series, one after another.
        data = raw.reshape(n_channels, -1).astype(float)
    else:
        raise NotImplementedError(f"Unknown data orientation {orientation!r}")

    # Apply each channel's resolution, then convert its unit to microvolts.
    for index in range(n_channels):
        factor, _ = _unit_factor(units[index])
        data[index] *= resolutions[index] * factor

    markers = _read_vmrk(vhdr_path.parent / common.get("MarkerFile", ""), fs)
    return data, names, fs, markers


def _unit_factor(unit: str) -> tuple[float, str]:
    """Multiplier taking ``unit`` to microvolts, plus a note about the choice."""
    key = str(unit).strip().lower()
    if key in _UNIT_TO_MICROVOLTS:
        return _UNIT_TO_MICROVOLTS[key], f"declared '{unit}'"
    warnings.warn(
        f"Unrecognised unit {unit!r}; treating it as microvolts.",
        stacklevel=3,
    )
    return 1.0, f"unrecognised unit {unit!r}, assumed microvolts"


def _read_vmrk(vmrk_path: Path, fs: float) -> list[Marker]:
    """Read markers from a BrainVision ``.vmrk`` file.

    Each entry looks like ``Mk2=Stimulus,S  1,1001,1,0``: the type, a
    description, the **onset in samples** (one-based), a duration in samples,
    and a channel number where 0 means all channels.

    In a BIDS dataset this file is usually near-empty, because the events live
    in ``events.tsv`` instead. It is read anyway so a plain BrainVision
    recording outside BIDS still yields its markers.
    """
    if not vmrk_path or not vmrk_path.is_file():
        return []

    markers: list[Marker] = []
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    try:
        # Same magic first line as the header, so skip to the first section.
        text = vmrk_path.read_text(encoding="utf-8", errors="replace")
        first_section = text.find("[")
        if first_section < 0:
            return []
        parser.read_string(text[first_section:], source=str(vmrk_path))
    except (configparser.Error, OSError):
        warnings.warn(f"Could not parse {vmrk_path.name}; ignoring its markers.",
                      stacklevel=2)
        return []

    if not parser.has_section("Marker Infos"):
        return []

    for _, value in parser["Marker Infos"].items():
        fields = [f.strip() for f in value.split(",")]
        if len(fields) < 4:
            continue
        kind, description, position, duration = fields[0], fields[1], fields[2], fields[3]
        # "New Segment" entries mark a recording boundary, not an experimental
        # event, and their description field holds a timestamp rather than a
        # label.
        if kind.lower().startswith("new segment"):
            continue
        try:
            # Positions are one-based sample indices.
            onset = (int(position) - 1) / fs
            span = int(duration) / fs
        except ValueError:
            continue
        markers.append(Marker(
            name=description or kind,
            onset=max(0.0, onset),
            duration=max(0.0, span),
        ))
    return markers


# --------------------------------------------------------------------------
# BIDS
# --------------------------------------------------------------------------

def find_bids_recordings(root: str | Path) -> list[Path]:
    """List every EEG recording in a BIDS dataset, as paths to its header.

    Walks ``sub-*/[ses-*/]eeg/`` looking for BrainVision headers. Returns them
    sorted, so the order is stable between runs.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(root.glob("sub-*/**/eeg/*_eeg.vhdr"))


def describe_bids_dataset(root: str | Path) -> dict:
    """Summarise a BIDS dataset: its name, its subjects, and what they contain.

    Reads ``dataset_description.json`` and ``participants.tsv`` if present.
    Neither is required for the data to be readable, so both are optional here.
    """
    root = Path(root).expanduser().resolve()
    description: dict = {"root": root, "name": None, "bids_version": None,
                         "authors": [], "license": None, "participants": [],
                         "recordings": []}

    description_path = root / "dataset_description.json"
    if description_path.is_file():
        meta = json.loads(description_path.read_text(encoding="utf-8"))
        description["name"] = meta.get("Name")
        description["bids_version"] = meta.get("BIDSVersion")
        description["authors"] = meta.get("Authors", [])
        description["license"] = meta.get("License")

    participants_path = root / "participants.tsv"
    if participants_path.is_file():
        description["participants"] = _read_tsv(participants_path)

    description["recordings"] = find_bids_recordings(root)
    return description


def read_bids_recording(vhdr_path: str | Path) -> Recording:
    """Read one BIDS EEG recording, with its channels and events.

    Three sidecars sit beside the signal, each adding something the binary file
    does not carry:

    * ``*_eeg.json`` -- the sampling rate, the reference electrode, the mains
      frequency, and what filtering was applied during recording.
    * ``*_channels.tsv`` -- one row per channel: its name, its **type** (EEG,
      EOG, TRIG and so on) and its units. The type matters: a trigger channel
      is not a signal, and averaging it into an occipital ROI would be a
      mistake.
    * ``*_events.tsv`` -- one row per event, with ``onset`` and ``duration``
      **in seconds** and usually a ``trial_type`` naming the condition. This is
      where BIDS keeps what a recorder would have put in its own marker file.

    Parameters
    ----------
    vhdr_path : path
        The ``.vhdr`` header of the recording.

    Returns
    -------
    Recording
        Signals in microvolts, with events as markers. Non-signal channels
        (triggers, and anything else the channels file does not call a
        biosignal) are kept in ``.digital`` rather than mixed into ``.data``.
    """
    vhdr_path = Path(vhdr_path).expanduser().resolve()
    data, names, fs, vmrk_markers = read_brainvision(vhdr_path)

    # The sidecars share the recording's stem with the "_eeg" suffix swapped.
    stem = vhdr_path.name
    for suffix in ("_eeg.vhdr", ".vhdr"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    folder = vhdr_path.parent

    unit_note = "microvolts (BrainVision resolution and units applied)"

    # --- the JSON sidecar: rate and provenance ---------------------------
    json_path = folder / f"{stem}_eeg.json"
    sidecar: dict = {}
    if json_path.is_file():
        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        declared_fs = sidecar.get("SamplingFrequency")
        if declared_fs and abs(float(declared_fs) - fs) / fs > 0.001:
            warnings.warn(
                f"{json_path.name} declares {declared_fs} Hz but the header "
                f"implies {fs:g} Hz. Using the header, which describes the file "
                "that actually holds the samples.",
                stacklevel=2,
            )

    # --- channels.tsv: which rows are signal, and which are not ----------
    channels_path = folder / f"{stem}_channels.tsv"
    signal_rows: list[int] = []
    other: dict[str, np.ndarray] = {}
    if channels_path.is_file():
        rows = _read_tsv(channels_path)
        if len(rows) != len(names):
            warnings.warn(
                f"{channels_path.name} has {len(rows)} rows but the recording "
                f"has {len(names)} channels; using the header's channel list.",
                stacklevel=2,
            )
            signal_rows = list(range(len(names)))
        else:
            for index, row in enumerate(rows):
                kind = (row.get("type") or "").strip().upper()
                # Anything that is not a biosignal is set aside rather than
                # dropped: a trigger channel is often exactly what you need to
                # check the events against, but it must never be averaged in.
                if kind in ("TRIG", "MISC", "SYSCLOCK", "ADC", "DAC", "OTHER"):
                    other[names[index]] = data[index]
                else:
                    signal_rows.append(index)
    else:
        signal_rows = list(range(len(names)))

    # --- events.tsv: the experiment's structure --------------------------
    events_path = folder / f"{stem}_events.tsv"
    markers: list[Marker] = []
    if events_path.is_file():
        for row in _read_tsv(events_path):
            onset = _as_float(row.get("onset"))
            if not np.isfinite(onset):
                continue
            # BIDS says duration is seconds, and n/a when unknown. An unknown
            # duration is an instant as far as this reader is concerned.
            duration = _as_float(row.get("duration"), default=0.0)
            # trial_type is the conventional label; fall back to value, then to
            # a generic name, so an events file without trial_type still works.
            name = (row.get("trial_type") or row.get("value") or "event").strip()
            payload = {k: v for k, v in row.items()
                       if k not in ("onset", "duration", "trial_type") and v not in ("", "n/a")}
            markers.append(Marker(
                name=name or "event",
                onset=onset,
                duration=max(0.0, duration),
                payload=payload or None,
            ))
        markers.sort(key=lambda m: m.onset)
    else:
        # No events file: fall back to whatever the BrainVision markers hold.
        markers = vmrk_markers

    return Recording(
        data=np.ascontiguousarray(data[signal_rows, :]),
        channels=[names[i] for i in signal_rows],
        fs=fs,
        markers=markers,
        unit_note=unit_note,
        source=vhdr_path,
        digital=other,
    )
