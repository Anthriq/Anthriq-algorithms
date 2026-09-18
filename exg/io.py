# Copyright 2026 Anthriq
# Licensed under the Apache License, Version 2.0. See the LICENSE file.

"""
Loading biosignal recordings, and the markers that go with them.

Two layouts are supported, because the two ways you are likely to get data off
an Anthriq front end look quite different.

--------------------------------------------------------------------------------
Layout 1: a BXI Studio export
--------------------------------------------------------------------------------
Exporting a recording writes a *folder*, not a file::

    my_recording/
    |-- meta.json            <- sample rate, channel labels, units, MARKERS
    `-- my_recording.csv     <- one row per sample

The CSV header is the channel names you typed during setup, followed by a
``timestamp`` column::

    O1,O2,Fpz,timestamp
    12.5,-3.25,0.75,1718000000123456

Two things routinely catch people out:

* ``timestamp`` is in **microseconds**, as a large integer -- not seconds. Divide
  a difference by 1e6, not by 1.
* **The markers are not in the CSV.** They live in ``meta.json``, in a compact
  encoding described under :func:`decode_markers`. Analysing the CSV alone loses
  every event.

--------------------------------------------------------------------------------
Layout 2: a raw DAQ capture
--------------------------------------------------------------------------------
Recording straight from the data-acquisition device, bypassing BXI Studio, gives
a plain CSV whose columns are named after the DAQ's own terminals::

    Mod_9234/ai0,Mod_9234/ai1,Mod_9401/port0/line0
    0.00123,-0.00047,0

or the same thing with bare names (``ai0``, ``port0/line0``). Here:

* ``ai*`` columns are analogue signals, in **volts at the converter**. Divide by
  the amplifier gain to recover the voltage at the electrode.
* ``port*/line*`` columns are digital marker lines. They carry a **brief pulse at
  each event**, not a level held for the whole epoch, so events are found as
  rising edges (see :func:`find_rising_edges`).
* There is **no timestamp column**, so the sampling rate cannot be recovered from
  the file. You must supply it.

--------------------------------------------------------------------------------
Units
--------------------------------------------------------------------------------
Both loaders return signals in **microvolts**, always, so that
:mod:`exg.spectra` gives uV**2 band powers without further thought. Whatever
scaling was applied is recorded in ``Recording.unit_note`` so you can check it.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "Recording",
    "Marker",
    "load",
    "load_bxi_export",
    "load_daq_csv",
    "decode_markers",
    "find_rising_edges",
]

# Multiply by these to get microvolts.
_UNIT_TO_MICROVOLTS = {
    "v": 1e6,
    "volt": 1e6,
    "volts": 1e6,
    "mv": 1e3,
    "millivolt": 1e3,
    "uv": 1.0,
    "µv": 1.0,  # micro sign
    "μv": 1.0,  # Greek mu
    "microvolt": 1.0,
    "microvolts": 1.0,
    "nv": 1e-3,
}

# Column names in a raw DAQ capture. Tolerates an optional module prefix, so
# both "Mod_9234/ai0" and "ai0" match, likewise "Mod_9401/port0/line1".
_ANALOG_COLUMN = re.compile(r"^(?:[A-Za-z0-9_]+/)?ai\d+$", re.IGNORECASE)
_DIGITAL_COLUMN = re.compile(r"^(?:[A-Za-z0-9_]+/)?port\d+/line\d+$", re.IGNORECASE)


@dataclass
class Marker:
    """One event on the recording's timeline.

    Attributes
    ----------
    name : str
        Label, as it was set when recording.
    onset : float
        Seconds from the start of the recording.
    duration : float
        Seconds. ``0.0`` for an instant ("event"); positive for a span ("epoch").
    payload : dict or None
        Any extra data attached to this occurrence.
    """

    name: str
    onset: float
    duration: float = 0.0
    payload: dict | None = None

    @property
    def offset(self) -> float:
        """When the marker ends, in seconds."""
        return self.onset + self.duration


@dataclass
class Recording:
    """A loaded recording: signals, channel names, sampling rate, markers.

    Attributes
    ----------
    data : ndarray
        Shape (n_channels, n_samples), in **microvolts**.
    channels : list of str
        Channel names, in the same order as ``data``'s rows.
    fs : float
        Sampling rate in Hz.
    markers : list of Marker
        Events, ordered by onset. Empty if the recording has none.
    unit_note : str
        How the data came to be in microvolts. Worth printing when a result
        looks off by a power of ten.
    source : Path or None
        Where it was loaded from.
    digital : dict
        Any raw digital marker lines, by column name. Only for DAQ captures.
    """

    data: np.ndarray
    channels: list[str]
    fs: float
    markers: list[Marker] = field(default_factory=list)
    unit_note: str = ""
    source: Path | None = None
    digital: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        """Length of the recording in seconds."""
        return self.n_samples / self.fs

    def pick(self, names: list[str] | tuple[str, ...]) -> np.ndarray:
        """Return the rows for the named channels, in the order given.

        Matching ignores case and surrounding whitespace, because a channel
        typed as "o1 " during setup should still be findable as "O1".

        Raises
        ------
        KeyError
            If any name is absent. The message lists what is available, since
            channel names are chosen by whoever recorded the data and cannot be
            guessed from the outside.
        """
        lookup = {c.strip().lower(): i for i, c in enumerate(self.channels)}
        rows, missing = [], []
        for name in names:
            index = lookup.get(name.strip().lower())
            if index is None:
                missing.append(name)
            else:
                rows.append(index)
        if missing:
            raise KeyError(
                f"Channel(s) not in this recording: {missing}. "
                f"Available: {self.channels}"
            )
        return self.data[rows, :]

    def markers_named(self, name: str) -> list[Marker]:
        """Every marker whose name matches ``name``, ignoring case."""
        target = name.strip().lower()
        return [m for m in self.markers if m.name.strip().lower() == target]

    def marker_names(self) -> list[str]:
        """The distinct marker names present, in order of first appearance."""
        seen: list[str] = []
        for marker in self.markers:
            if marker.name not in seen:
                seen.append(marker.name)
        return seen

    def summary(self) -> str:
        """A short human-readable description, for printing after a load."""
        lines = [
            f"  Source     : {self.source.name if self.source else '(in memory)'}",
            f"  Channels   : {len(self.channels)} -> {', '.join(self.channels)}",
            f"  Sample rate: {self.fs:g} Hz",
            f"  Duration   : {self.duration:.1f} s ({self.n_samples} samples)",
            f"  Units      : {self.unit_note}",
        ]
        if self.markers:
            counts = {n: len(self.markers_named(n)) for n in self.marker_names()}
            detail = ", ".join(f"{n} x{c}" for n, c in counts.items())
            lines.append(f"  Markers    : {len(self.markers)} -> {detail}")
        else:
            lines.append("  Markers    : none")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

def _scale_to_microvolts(unit: str | None) -> tuple[float, str]:
    """Return (multiplier, description) taking ``unit`` to microvolts."""
    if unit is None:
        return 1.0, "assumed microvolts (no unit declared)"
    key = str(unit).strip().lower()
    if key in _UNIT_TO_MICROVOLTS:
        factor = _UNIT_TO_MICROVOLTS[key]
        if factor == 1.0:
            return 1.0, f"microvolts (declared '{unit}')"
        return factor, f"converted from '{unit}' to microvolts (x{factor:g})"
    warnings.warn(
        f"Unrecognised unit {unit!r}; treating the data as microvolts. "
        f"Known units: {sorted(_UNIT_TO_MICROVOLTS)}",
        stacklevel=3,
    )
    return 1.0, f"assumed microvolts (unrecognised unit {unit!r})"


def _sanity_check_amplitude(data: np.ndarray, unit_note: str) -> None:
    """Warn if the data does not look like microvolts after scaling.

    Scalp EEG is tens of microvolts, surface EMG up to a few thousand. A median
    absolute value far outside that range usually means the declared unit was
    wrong, which is worth flagging: every frequency-domain result would be off
    by the same factor and nothing else would look broken.

    Not appropriate for every recording. A bench measurement deliberately drives
    volts rather than microvolts, so pass ``expect_physiological=False`` to the
    loader for those and the check is skipped rather than crying wolf.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        warnings.warn("Recording contains no finite samples.", stacklevel=3)
        return
    median = float(np.median(np.abs(finite[finite != 0]))) if np.any(finite != 0) else 0.0
    if median == 0.0:
        return
    if median < 1e-3:
        warnings.warn(
            f"Median absolute amplitude is {median:.3g} uV, far below anything "
            f"physiological. The unit may be wrong ({unit_note}); volts read as "
            "microvolts would look like this.",
            stacklevel=3,
        )
    elif median > 1e5:
        warnings.warn(
            f"Median absolute amplitude is {median:.3g} uV (~{median / 1e6:.3g} V), "
            f"far above anything physiological. The unit may be wrong ({unit_note}).",
            stacklevel=3,
        )


# --------------------------------------------------------------------------
# Markers in a BXI export
# --------------------------------------------------------------------------

def decode_markers(meta: dict) -> list[Marker]:
    """Decode the marker block of a BXI ``meta.json``.

    The encoding
    ------------
    Markers are stored as a dictionary plus a delta-encoded sequence, which is
    compact when a run has many occurrences of a few kinds::

        {
          "defs": [ {"id": 0, "name": "stimulus", "kind": "event"},
                    {"id": 1, "name": "rest",     "kind": "epoch"} ],
          "seq":  { "def":  [0, 0, 1],
                    "dt":   [500000, 250000, 250000],
                    "dur":  [0, 0, 2000000],
                    "data": {"2": {"repetition": 3}} }
        }

    ``defs`` is the legend, written once per *kind* of marker. ``seq`` holds one
    entry per *occurrence*, in parallel arrays.

    Three details that are easy to get wrong, and each of which silently
    produces plausible-looking nonsense:

    1. **``dt`` is a gap, not a time.** Each value is the microseconds since the
       *previous* occurrence, so absolute times come from a running sum. Reading
       ``dt`` as absolute gives markers bunched at the start of the recording.
       The example above decodes to 0.5 s, 0.75 s and 1.0 s -- not 0.5, 0.25,
       0.25.
    2. **``def`` indexes by ``id``, not by position** in ``defs``.
    3. **``data`` is keyed by the occurrence's index written as a string.**
       Looking it up with an integer finds nothing, every time, with no error.

    Times are microseconds from the start of the recording (the file states this
    as ``timeBase: "us"`` and ``origin: "recordingStart"``). Both are honoured
    rather than assumed -- a marker set anchored to the wrong origin will place
    stimulus windows on rest periods, which inverts the result being measured
    rather than merely degrading it.

    Parameters
    ----------
    meta : dict
        The parsed ``meta.json``.

    Returns
    -------
    list of Marker
        Empty if the file carries no markers, which is not an error.
    """
    block = meta.get("markers", meta)
    definitions = block.get("defs")
    sequence = block.get("seq")
    if not definitions or not sequence:
        return []

    time_base = str(block.get("timeBase", "us")).lower()
    if time_base in ("us", "microseconds", "microsecond"):
        to_seconds = 1e-6
    elif time_base in ("ms", "milliseconds", "millisecond"):
        to_seconds = 1e-3
    elif time_base in ("s", "seconds", "second"):
        to_seconds = 1.0
    else:
        warnings.warn(
            f"Unknown marker timeBase {time_base!r}; assuming microseconds.",
            stacklevel=2,
        )
        to_seconds = 1e-6

    origin = str(block.get("origin", "recordingStart"))
    if origin != "recordingStart":
        warnings.warn(
            f"Marker origin is {origin!r}, not 'recordingStart'. Onsets are "
            "returned as stored and may need re-anchoring before use.",
            stacklevel=2,
        )

    # Detail 2: index the legend by its declared id.
    legend = {int(d["id"]): d for d in definitions}

    def_ids = sequence.get("def", [])
    deltas = sequence.get("dt", [])
    durations = sequence.get("dur", [])
    payloads = sequence.get("data", {}) or {}

    markers: list[Marker] = []
    running_us = 0.0
    for index, def_id in enumerate(def_ids):
        # Detail 1: accumulate the gaps to recover absolute time.
        running_us += float(deltas[index]) if index < len(deltas) else 0.0

        definition = legend.get(int(def_id))
        if definition is None:
            warnings.warn(
                f"Marker occurrence {index} references unknown definition id "
                f"{def_id}; skipping it.",
                stacklevel=2,
            )
            continue

        duration_us = float(durations[index]) if index < len(durations) else 0.0
        markers.append(
            Marker(
                name=str(definition.get("name", f"marker_{def_id}")),
                onset=running_us * to_seconds,
                duration=duration_us * to_seconds,
                # Detail 3: the key is the index as a string.
                payload=payloads.get(str(index)),
            )
        )
    return markers


# --------------------------------------------------------------------------
# Markers on a digital line
# --------------------------------------------------------------------------

def find_rising_edges(line: np.ndarray, fs: float, *, min_gap: float = 0.01) -> np.ndarray:
    """Onset times, in seconds, of each low-to-high transition on a digital line.

    Why edges and not levels
    ------------------------
    The marker lines carry a **brief pulse** at each event -- often just a few
    samples -- rather than a level held for the duration of the epoch. Testing
    ``line != 0`` therefore finds every sample *within* each pulse, so a 3-sample
    pulse becomes 3 events. Taking the rising edge finds one event per pulse,
    which is what the hardware meant.

    Because a pulse is an onset only, the *duration* of the epoch it starts is
    not in the signal. It has to come from the protocol -- see the
    ``epoch_duration`` arguments on the analysis scripts.

    Parameters
    ----------
    line : ndarray
        The digital channel. Any numeric type; thresholded at its midpoint, so
        both 0/1 logic and 0/5 V TTL work without configuration.
    fs : float
        Sampling rate in Hz.
    min_gap : float
        Ignore an edge that follows the previous one by less than this many
        seconds. Suppresses contact bounce and ringing on a noisy line.
        Default 0.01 (10 ms).

    Returns
    -------
    ndarray
        Onset times in seconds, ascending. Empty if the line never goes high.
    """
    line = np.asarray(line, dtype=float).ravel()
    low, high = float(np.nanmin(line)), float(np.nanmax(line))

    # A line that never changes carries no events. Without this check, the
    # midpoint threshold on a constant line is the line itself and every sample
    # reads as "high".
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-12:
        return np.array([], dtype=float)

    above = line > (low + high) / 2.0
    edges = np.flatnonzero(np.diff(above.astype(np.int8)) == 1) + 1

    if edges.size == 0:
        return np.array([], dtype=float)

    times = edges / fs
    kept = [times[0]]
    for t in times[1:]:
        if t - kept[-1] >= min_gap:
            kept.append(t)
    return np.asarray(kept, dtype=float)


# --------------------------------------------------------------------------
# Reading CSVs
# --------------------------------------------------------------------------

def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a numeric CSV with a single header row. Returns (header, values)."""
    with open(path, newline="") as handle:
        first = handle.readline()
    if not first.strip():
        raise ValueError(f"{path} is empty")
    header = [h.strip() for h in first.rstrip("\r\n").split(",")]

    values = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    if values.ndim == 1:
        # A single-row or single-column file; make it 2-D either way.
        values = values.reshape(1, -1) if len(header) > 1 else values.reshape(-1, 1)
    if values.shape[1] != len(header):
        raise ValueError(
            f"{path}: header has {len(header)} columns but the data has "
            f"{values.shape[1]}."
        )
    return header, values


def _fs_from_timestamps(timestamps: np.ndarray) -> tuple[float, int]:
    """Infer the sampling rate from a microsecond timestamp column.

    Returns (fs, n_gaps). ``n_gaps`` counts intervals more than 50% longer than
    the median, which indicates dropped samples -- worth surfacing, because a
    gap makes the recording shorter in time than its sample count implies.
    """
    diffs = np.diff(timestamps.astype(float))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        raise ValueError(
            "The timestamp column never increases, so the sampling rate cannot "
            "be inferred from it. Pass the rate explicitly."
        )

    median_us = float(np.median(positive))
    # The column is in microseconds, so a 1 kHz recording steps by 1000.
    fs = 1e6 / median_us
    n_gaps = int(np.count_nonzero(diffs > 1.5 * median_us))
    return fs, n_gaps


# --------------------------------------------------------------------------
# The two loaders
# --------------------------------------------------------------------------

def load_bxi_export(folder: str | Path, *, unit: str | None = None) -> Recording:
    """Load a BXI Studio export folder (``meta.json`` plus a CSV).

    The sample rate comes from ``meta.json``, cross-checked against the
    ``timestamp`` column; a disagreement of more than 1% is reported, and the
    declared rate wins. Channel names come from ``meta.json`` when it lists
    them, otherwise from the CSV header.

    Parameters
    ----------
    folder : path
        The exported folder.
    unit : str, optional
        Override the unit declared in ``meta.json`` (``"uV"``, ``"mV"``, ``"V"``).

    Returns
    -------
    Recording
        Signals in microvolts, with markers decoded from the sidecar.
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder")

    meta_path = folder / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"No meta.json in {folder}. A BXI export contains meta.json next to "
            "the CSV; without it there is no sample rate and no markers. If you "
            "have only a bare CSV, load it with load_daq_csv(..., fs=...)."
        )
    meta = json.loads(meta_path.read_text())

    candidates = sorted(p for p in folder.glob("*.csv") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No CSV file in {folder}")
    if len(candidates) > 1:
        warnings.warn(
            f"{folder} holds {len(candidates)} CSV files; reading "
            f"{candidates[0].name}.",
            stacklevel=2,
        )
    csv_path = candidates[0]

    header, values = _read_csv(csv_path)

    # Everything except the timestamp column is a signal. Selecting positionally
    # rather than by pattern matters: a channel the operator happened to name
    # "Event" or "Trigger" is still a channel, and a name-based filter would
    # quietly drop it.
    if header and header[-1].lower() == "timestamp":
        signal_columns = list(range(len(header) - 1))
        timestamps = values[:, -1]
    else:
        warnings.warn(
            f"{csv_path.name} has no trailing 'timestamp' column; treating every "
            "column as a signal.",
            stacklevel=2,
        )
        signal_columns = list(range(len(header)))
        timestamps = None

    csv_names = [header[i] for i in signal_columns]

    # meta.json is the better source for names and units, but it is allowed to
    # list no channels even while giving numChannels, so fall back in steps.
    declared = meta.get("channels") or []
    if declared and len(declared) == len(signal_columns):
        channels = [str(c.get("label") or c.get("id") or f"ch{i + 1}")
                    for i, c in enumerate(declared)]
        declared_unit = declared[0].get("unit")
    else:
        if declared:
            warnings.warn(
                f"meta.json lists {len(declared)} channels but the CSV has "
                f"{len(signal_columns)}; using the CSV header for names.",
                stacklevel=2,
            )
        channels = csv_names or [f"ch{i + 1}" for i in signal_columns]
        declared_unit = None

    # A recording with numChannels but an empty channels array is documented as
    # possible; the exporter itself falls back to ch1..chN and microvolts.
    if not channels:
        n = int(meta.get("numChannels", len(signal_columns)))
        channels = [f"ch{i + 1}" for i in range(n)]

    # Sample rate: prefer the declared value, but check it.
    declared_fs = meta.get("sampleRate")
    fs: float | None = float(declared_fs) if declared_fs else None
    n_gaps = 0
    if timestamps is not None and timestamps.size > 1:
        inferred_fs, n_gaps = _fs_from_timestamps(timestamps)
        if fs is None:
            fs = inferred_fs
        elif abs(inferred_fs - fs) / fs > 0.01:
            warnings.warn(
                f"meta.json declares {fs:g} Hz but the timestamps imply "
                f"{inferred_fs:g} Hz. Using the declared rate. A mismatch this "
                "large usually means dropped samples.",
                stacklevel=2,
            )
    if fs is None:
        raise ValueError(
            f"{meta_path} has no 'sampleRate' and the CSV has no usable "
            "timestamp column, so the sampling rate is unknown. Pass it "
            "explicitly."
        )

    if n_gaps:
        warnings.warn(
            f"{n_gaps} gap(s) in the timestamps: some samples were dropped "
            "during recording. Times derived from sample counts will drift "
            "past each gap.",
            stacklevel=2,
        )

    factor, unit_note = _scale_to_microvolts(unit or declared_unit or "uV")
    data = values[:, signal_columns].T * factor
    _sanity_check_amplitude(data, unit_note)

    return Recording(
        data=np.ascontiguousarray(data),
        channels=channels,
        fs=float(fs),
        markers=decode_markers(meta),
        unit_note=unit_note,
        source=csv_path,
    )


def load_daq_csv(
    path: str | Path,
    *,
    fs: float,
    gain: float = 1.0,
    unit: str = "V",
    epoch_durations: dict[str, float] | None = None,
    marker_names: dict[str, str] | None = None,
    expect_physiological: bool = True,
) -> Recording:
    """Load a raw DAQ capture: a plain CSV of analogue and digital columns.

    Parameters
    ----------
    path : path
        The CSV file.
    fs : float
        Sampling rate in Hz. **Required** -- the file carries no timing
        information, and a wrong rate silently rescales every frequency in every
        result, so there is deliberately no default.
    gain : float
        Total amplifier gain between electrode and converter. The analogue
        columns are divided by it to recover the voltage at the electrode. Left
        at 1.0, amplitudes are converter-referred and a note records that.
    unit : str
        Unit of the analogue columns before the gain division. Raw DAQ captures
        are in volts, which is the default.
    epoch_durations : dict, optional
        Seconds per epoch, keyed by digital column name, e.g.
        ``{"port0/line0": 10.0}``. A digital pulse marks only an *onset*, so the
        duration has to come from the protocol. Without it, markers are instants.
    marker_names : dict, optional
        Friendlier names for digital columns, e.g.
        ``{"port0/line0": "stimulus"}``. Matching ignores any module prefix.
    expect_physiological : bool
        Whether to warn when amplitudes are far outside the physiological range.
        Set False for a bench capture, which drives volts on purpose. Default True.

    Returns
    -------
    Recording
        Analogue signals in microvolts; digital lines in ``.digital`` and decoded
        into ``.markers``.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if fs is None or fs <= 0:
        raise ValueError("fs must be a positive sampling rate in Hz")
    if gain <= 0:
        raise ValueError("gain must be positive")

    header, values = _read_csv(path)

    analog = [i for i, name in enumerate(header) if _ANALOG_COLUMN.match(name)]
    digital = [i for i, name in enumerate(header) if _DIGITAL_COLUMN.match(name)]

    # Anything unrecognised is treated as analogue rather than discarded: a
    # capture written with human channel names is more useful loaded than
    # refused.
    other = [i for i in range(len(header)) if i not in analog and i not in digital]
    if other and not analog:
        analog = other
        other = []
    if other:
        warnings.warn(
            f"Ignoring unrecognised column(s): {[header[i] for i in other]}",
            stacklevel=2,
        )

    if not analog:
        raise ValueError(
            f"No analogue columns found in {path.name}. Expected names like "
            f"'ai0' or 'Mod_9234/ai0'. Header was: {header}"
        )

    factor, unit_note = _scale_to_microvolts(unit)
    data = values[:, analog].T * factor / gain
    if gain == 1.0:
        unit_note += "; no gain applied, so amplitudes are converter-referred"
    else:
        unit_note += f"; divided by a gain of {gain:g} to reach the electrode"
    if expect_physiological:
        _sanity_check_amplitude(data, unit_note)

    # Digital lines -> markers, one per rising edge.
    def strip_prefix(name: str) -> str:
        return name.split("/", 1)[1] if name.lower().startswith("mod_") else name

    epoch_durations = {strip_prefix(k): v for k, v in (epoch_durations or {}).items()}
    marker_names = {strip_prefix(k): v for k, v in (marker_names or {}).items()}

    digital_lines: dict[str, np.ndarray] = {}
    markers: list[Marker] = []
    for column in digital:
        raw_name = header[column]
        key = strip_prefix(raw_name)
        line = values[:, column]
        digital_lines[raw_name] = line

        duration = float(epoch_durations.get(key, 0.0))
        label = marker_names.get(key, key)
        for onset in find_rising_edges(line, fs):
            markers.append(Marker(name=label, onset=float(onset), duration=duration))

    markers.sort(key=lambda m: m.onset)

    return Recording(
        data=np.ascontiguousarray(data),
        channels=[header[i] for i in analog],
        fs=float(fs),
        markers=markers,
        unit_note=unit_note,
        source=path,
        digital=digital_lines,
    )


def load(path: str | Path, *, fs: float | None = None, **kwargs) -> Recording:
    """Load a recording, choosing the right reader for what is on disk.

    The choice is made from the *structure* of the input, never from its name:

    * a folder containing ``meta.json``  -> :func:`load_bxi_export`
    * a CSV with a sidecar ``meta.json`` -> :func:`load_bxi_export` on its folder
    * any other CSV                      -> :func:`load_daq_csv` (needs ``fs``)

    Parameters
    ----------
    path : path
        A BXI export folder, or a CSV file.
    fs : float, optional
        Sampling rate, required for a bare CSV and ignored for a BXI export
        (which declares its own).
    **kwargs
        Passed through to the chosen reader.

    Returns
    -------
    Recording
    """
    path = Path(path).expanduser().resolve()

    if path.is_dir():
        if not (path / "meta.json").is_file():
            csvs = sorted(path.glob("*.csv"))
            raise FileNotFoundError(
                f"{path} is a folder with no meta.json, so it is not a BXI "
                f"export. It holds {len(csvs)} CSV file(s); load one directly "
                "with an explicit fs, or point at the export folder."
            )
        return load_bxi_export(path, **kwargs)

    if not path.is_file():
        raise FileNotFoundError(path)

    if (path.parent / "meta.json").is_file():
        return load_bxi_export(path.parent, **kwargs)

    if fs is None:
        raise ValueError(
            f"{path.name} is a bare CSV with no meta.json beside it, so its "
            "sampling rate is unknown. Pass fs=<rate in Hz>. Guessing it would "
            "silently rescale every frequency in the results."
        )
    return load_daq_csv(path, fs=fs, **kwargs)
