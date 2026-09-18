#!/usr/bin/env python3
# Copyright 2026 Nexstem India Private Limited (trading as Anthriq)
# Licensed under the Apache License, Version 2.0. See the LICENSE file.
"""
Alpha reactivity: the Berger effect.

Close your eyes and a rhythm near 10 Hz appears over the back of the head. Open
them and it largely disappears. Hans Berger saw this in 1929 in the first human
EEG recording ever made, and it is still the most reliable thing you can measure
with a few electrodes and ten minutes.

This script compares the eyes-closed and eyes-open conditions of one recording
and reports:

  * band power in the alpha range, per condition
  * the reactivity ratio (closed / open)
  * a normalised reactivity index, bounded in [-1, +1]
  * individual peak alpha frequency

Usage
-----
    # A BXI Studio export, with eyes_closed / eyes_open markers
    python scripts/alpha.py my_recording/ --sites O1,O2

    # The same, but the conditions were not marked, so give time windows
    python scripts/alpha.py my_recording/ --sites O1,O2 \\
        --closed-window 10,40 --open-window 45,60

    # A bare CSV straight off the DAQ: the sample rate must be supplied
    python scripts/alpha.py bench.csv --fs 1000 --gain 2000 --sites ai0,ai1 \\
        --closed-window 10,40 --open-window 45,60

    # No hardware yet? Generate a recording with a known answer and analyse it
    python scripts/synth.py alpha /tmp/demo && python scripts/alpha.py /tmp/demo --sites O1,O2

Why the montage matters
-----------------------
Alpha is generated in and around the visual cortex, so it is strongest at the
occipital sites O1 and O2 at the back of the head, and much weaker frontally. A
frontal channel such as Fpz therefore makes a useful control: if your "alpha"
is just as large at the forehead, you are probably looking at an artefact rather
than at a cortical rhythm.

References
----------
Berger, H. (1929). Über das Elektrenkephalogramm des Menschen. Archiv für
    Psychiatrie und Nervenkrankheiten, 87, 527-570.
Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and
    memory performance. Brain Research Reviews, 29(2-3), 169-195.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make `exg` importable when this file is run directly from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exg.io import Recording, load  # noqa: E402
from exg.plotting import save_figure, style_axes  # noqa: E402
from exg.spectra import band_power, peak_in_band, welch_psd  # noqa: E402

# The alpha band, 8-13 Hz, as it is defined in the literature. Some sources
# narrow it to 8-12, which shifts the computed power slightly, so whichever you
# use, report it alongside the number. Change it with --band.
DEFAULT_ALPHA_BAND = (8.0, 13.0)

# Search a little wider than the band when hunting for the individual peak, so
# a genuine peak near an edge is found and reported rather than clipped to it.
DEFAULT_PEAK_SEARCH = (7.0, 14.0)

# A peak this far above the local spectrum is worth calling a peak. Below it,
# what you have is the ordinary 1/f slope of the background, not a rhythm.
MIN_PEAK_PROMINENCE_DB = 3.0

# A closed-to-open power ratio at or above this is a convincing Berger effect.
# Ratios of 2x or more are common, and much larger ones are routine in a
# relaxed subject with good occipital contact. This describes the phenomenon
# rather than any particular amplifier, so it travels between setups.
CLEAR_REACTIVITY_RATIO = 2.0


def collect_condition(
    recording: Recording,
    rows: np.ndarray,
    *,
    marker_name: str | None,
    window: tuple[float, float] | None,
) -> np.ndarray:
    """Gather the samples belonging to one condition.

    Two ways to say which samples belong to a condition:

    * **By marker.** If the recording carries markers, every occurrence with a
      matching name contributes its span. Spans are concatenated, which is fine
      for a power spectrum: Welch averages over segments anyway, and the joins
      add a handful of discontinuities among thousands of samples.
    * **By time window.** If nothing was marked, give start and end seconds.

    Parameters
    ----------
    recording : Recording
    rows : ndarray
        Already-selected channel rows, shape (n_channels, n_samples).
    marker_name : str or None
        Marker to match. Tried first when the recording has markers.
    window : (float, float) or None
        Fallback start and end in seconds.

    Returns
    -------
    ndarray
        Shape (n_channels, n_selected_samples).
    """
    if marker_name:
        matches = recording.markers_named(marker_name)
        if matches:
            pieces = []
            for marker in matches:
                start = int(round(marker.onset * recording.fs))
                # A marker with no duration marks an instant, not a span, so it
                # cannot delimit a condition on its own.
                if marker.duration <= 0:
                    continue
                stop = int(round(marker.offset * recording.fs))
                piece = rows[:, max(0, start):min(rows.shape[1], stop)]
                if piece.shape[1] > 0:
                    pieces.append(piece)
            if pieces:
                return np.concatenate(pieces, axis=1)

    if window is None:
        raise ValueError(
            f"Could not find samples for the '{marker_name}' condition. The "
            f"recording's markers are {recording.marker_names() or 'none'}. "
            "Either pass a marker name that exists, or give an explicit time "
            "window such as --closed-window 10,40."
        )

    start, stop = window
    if stop <= start:
        raise ValueError(f"Window end ({stop}) must be after its start ({start})")
    i0 = int(round(start * recording.fs))
    i1 = int(round(stop * recording.fs))
    if i0 >= rows.shape[1]:
        raise ValueError(
            f"Window starts at {start} s but the recording is only "
            f"{recording.duration:.1f} s long."
        )
    return rows[:, max(0, i0):min(rows.shape[1], i1)]


def analyse(
    recording: Recording,
    sites: list[str],
    *,
    alpha_band: tuple[float, float] = DEFAULT_ALPHA_BAND,
    peak_search: tuple[float, float] = DEFAULT_PEAK_SEARCH,
    closed_marker: str = "eyes_closed",
    open_marker: str = "eyes_open",
    closed_window: tuple[float, float] | None = None,
    open_window: tuple[float, float] | None = None,
    window_seconds: float | None = None,
) -> dict:
    """Compare alpha power between eyes-closed and eyes-open.

    Returns a dict of results; every key carries its unit in the name so a
    number can never be mistaken for a different quantity.
    """
    rows = recording.pick(sites)

    closed = collect_condition(
        recording, rows, marker_name=closed_marker, window=closed_window
    )
    opened = collect_condition(
        recording, rows, marker_name=open_marker, window=open_window
    )

    # Use one window length for both conditions. This is not a detail: the PSD
    # is a density, so it is comparable across window lengths in principle, but
    # the *variance* of the estimate is not, and neither is the frequency
    # resolution. Comparing a 4 s-window spectrum against a 1 s-window spectrum
    # invites reading a resolution difference as a physiological one.
    shortest = min(closed.shape[1], opened.shape[1])
    nperseg = int(window_seconds * recording.fs) if window_seconds else None
    if nperseg and nperseg > shortest:
        raise ValueError(
            f"--window {window_seconds} s needs {nperseg} samples but the "
            f"shorter condition has only {shortest} "
            f"({shortest / recording.fs:.1f} s)."
        )

    # Average the chosen sites before computing the spectrum. Averaging in the
    # time domain suppresses noise that is independent between electrodes while
    # preserving signal that is common to them, which is what "occipital alpha"
    # conventionally means. Per-channel values are computed too, below, because
    # a single bad electrode is visible there and invisible in the average.
    freqs_c, psd_c = welch_psd(closed.mean(axis=0), recording.fs, nperseg=nperseg)
    freqs_o, psd_o = welch_psd(opened.mean(axis=0), recording.fs, nperseg=nperseg)

    power_closed = float(band_power(freqs_c, psd_c, *alpha_band))
    power_open = float(band_power(freqs_o, psd_o, *alpha_band))

    # Two ways to express the change, both worth having.
    #
    # The ratio is the intuitive one and the one most papers quote, but it is
    # unbounded and unstable when the denominator is small: a quiet eyes-open
    # condition can produce a ratio of 50 that means little.
    ratio = power_closed / power_open if power_open > 0 else float("inf")
    #
    # The normalised index is bounded in [-1, +1], degrades gracefully when one
    # condition is near zero, and is far better behaved when pooling across a
    # class where electrode impedance varies between subjects. Prefer it when
    # comparing people; quote the ratio when comparing with the literature.
    total = power_closed + power_open
    normalised = (power_closed - power_open) / total if total > 0 else float("nan")

    # Individual peak alpha frequency, from the eyes-closed spectrum where the
    # rhythm is strongest. It differs between people (typically 8-13 Hz), is
    # stable within a person across sessions, and drifts down slowly with age.
    peak_freq, peak_psd = peak_in_band(freqs_c, psd_c, *peak_search)

    # Is that peak real, or just the top of the 1/f background? Compare it with
    # the median of the search range: a genuine rhythm stands clear of it.
    local_median = float(np.median(psd_c[(freqs_c >= peak_search[0]) & (freqs_c <= peak_search[1])]))
    prominence_db = float(10.0 * np.log10(peak_psd / (local_median + 1e-30)))
    if prominence_db < MIN_PEAK_PROMINENCE_DB:
        peak_freq = float("nan")

    # Per-channel reactivity, to expose an electrode that is not contributing.
    per_channel = {}
    for index, name in enumerate(sites):
        _, ch_closed = welch_psd(closed[index], recording.fs, nperseg=nperseg)
        _, ch_open = welch_psd(opened[index], recording.fs, nperseg=nperseg)
        c = float(band_power(freqs_c, ch_closed, *alpha_band))
        o = float(band_power(freqs_o, ch_open, *alpha_band))
        per_channel[name] = {
            "closed_uv2": c,
            "open_uv2": o,
            "normalised": (c - o) / (c + o) if (c + o) > 0 else float("nan"),
        }

    return {
        "sites": sites,
        "alpha_band_hz": alpha_band,
        "alpha_power_closed_uv2": power_closed,
        "alpha_power_open_uv2": power_open,
        "reactivity_ratio": ratio,
        "reactivity_normalised": normalised,
        "peak_frequency_hz": peak_freq,
        "peak_prominence_db": prominence_db,
        "closed_seconds": closed.shape[1] / recording.fs,
        "open_seconds": opened.shape[1] / recording.fs,
        "frequency_resolution_hz": float(freqs_c[1] - freqs_c[0]),
        "per_channel": per_channel,
        "_spectra": (freqs_c, psd_c, freqs_o, psd_o),
    }


def plot(results: dict, output: Path, *, plot_band: tuple[float, float] = (1.0, 45.0)) -> Path:
    """Draw the spectra, the band powers, and the per-channel breakdown."""
    import matplotlib.pyplot as plt

    freqs_c, psd_c, freqs_o, psd_o = results["_spectra"]
    lo, hi = results["alpha_band_hz"]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel 1: the spectra. This is the panel that carries the result.
    ax = axes[0]
    mask_c = (freqs_c >= plot_band[0]) & (freqs_c <= plot_band[1])
    mask_o = (freqs_o >= plot_band[0]) & (freqs_o <= plot_band[1])
    ax.axvspan(lo, hi, color="0.9", zorder=0, label=f"alpha {lo:g}-{hi:g} Hz")
    ax.semilogy(freqs_c[mask_c], psd_c[mask_c], color="#D9531E", lw=1.6, label="eyes closed")
    ax.semilogy(freqs_o[mask_o], psd_o[mask_o], color="#1E7D8C", lw=1.6, label="eyes open")
    peak = results["peak_frequency_hz"]
    if np.isfinite(peak):
        ax.axvline(peak, color="#6E655A", ls="--", lw=1.0)
        ax.annotate(
            f"{peak:.2f} Hz",
            xy=(peak, ax.get_ylim()[1]),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=9,
            color="#6E655A",
        )
    style_axes(ax, xlabel="Frequency (Hz)", ylabel="PSD (uV$^2$/Hz)",
               title="Occipital spectrum by condition")
    ax.legend(frameon=False, fontsize=9)

    # Panel 2: the two band powers side by side, with the headline numbers.
    ax = axes[1]
    values = [results["alpha_power_closed_uv2"], results["alpha_power_open_uv2"]]
    ax.bar(["closed", "open"], values, color=["#D9531E", "#1E7D8C"], width=0.6)
    for x, value in enumerate(values):
        ax.text(x, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    style_axes(ax, ylabel=f"Alpha power {lo:g}-{hi:g} Hz (uV$^2$)",
               title=(f"ratio {results['reactivity_ratio']:.1f}x   "
                      f"index {results['reactivity_normalised']:+.2f}"))

    # Panel 3: per-channel, so a dead electrode is obvious.
    ax = axes[2]
    names = list(results["per_channel"])
    indices = [results["per_channel"][n]["normalised"] for n in names]
    ax.barh(names, indices, color="#6E655A", height=0.5)
    ax.axvline(0.0, color="0.3", lw=0.8)
    ax.set_xlim(-1.0, 1.0)
    style_axes(ax, xlabel="Normalised reactivity", title="Per channel")

    return save_figure(figure, output)


def report(results: dict) -> None:
    """Print the results, with enough context to interpret them."""
    lo, hi = results["alpha_band_hz"]
    print(f"\n  Sites               : {', '.join(results['sites'])}")
    print(f"  Alpha band          : {lo:g}-{hi:g} Hz")
    print(f"  Frequency resolution: {results['frequency_resolution_hz']:.3f} Hz")
    print(f"  Data analysed       : {results['closed_seconds']:.1f} s closed, "
          f"{results['open_seconds']:.1f} s open")
    print(f"\n  Alpha power, closed : {results['alpha_power_closed_uv2']:10.2f} uV^2")
    print(f"  Alpha power, open   : {results['alpha_power_open_uv2']:10.2f} uV^2")
    print(f"  Reactivity ratio    : {results['reactivity_ratio']:10.2f} x")
    print(f"  Reactivity index    : {results['reactivity_normalised']:+10.3f}   (-1 to +1)")

    peak = results["peak_frequency_hz"]
    if np.isfinite(peak):
        print(f"  Peak alpha frequency: {peak:10.2f} Hz "
              f"({results['peak_prominence_db']:.1f} dB above background)")
    else:
        print(f"  Peak alpha frequency:      none resolvable "
              f"({results['peak_prominence_db']:.1f} dB prominence, "
              f"below the {MIN_PEAK_PROMINENCE_DB:g} dB threshold)")

    ratio = results["reactivity_ratio"]
    print()
    if ratio >= CLEAR_REACTIVITY_RATIO:
        print(f"  A ratio above {CLEAR_REACTIVITY_RATIO:g}x is a clear result: alpha is "
              "suppressed by eye opening.")
    elif ratio > 1.2:
        print("  A modest effect in the right direction. More data, or better electrode")
        print("  contact, would firm it up.")
    elif ratio >= 0.8:
        print("  No reactivity worth reporting. Check electrode contact at the occipital")
        print("  sites first, then whether the subject was actually relaxed with their")
        print("  eyes closed: mental effort suppresses alpha as effectively as light does.")
    else:
        print("  Alpha is LARGER with the eyes open, which is backwards. The usual cause")
        print("  is the two conditions being swapped, so check the marker names or the")
        print("  window boundaries before looking for a physiological explanation.")


def parse_window(text: str | None) -> tuple[float, float] | None:
    """Parse a 'start,end' pair of seconds."""
    if not text:
        return None
    parts = text.replace(":", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected a window as 'start,end' in seconds, got {text!r}"
        )
    return (float(parts[0]), float(parts[1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure alpha reactivity between eyes-closed and eyes-open.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----")[1].split("Why the montage")[0],
    )
    parser.add_argument("input", help="A BXI export folder, or a CSV file.")
    parser.add_argument(
        "--sites",
        help="Channels to analyse, comma-separated, e.g. O1,O2. Required: the "
             "names are whatever was typed during setup, so they cannot be guessed.",
    )
    parser.add_argument("--fs", type=float, default=None,
                        help="Sample rate in Hz. Required for a bare CSV.")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Amplifier gain, for a raw DAQ capture. Default 1.")
    parser.add_argument("--band", default="8,13",
                        help="Alpha band edges in Hz. Default 8,13.")
    parser.add_argument("--peak-search", default="7,14",
                        help="Range to search for the individual peak. Default 7,14.")
    parser.add_argument("--closed-marker", default="eyes_closed",
                        help="Marker name for eyes-closed. Default eyes_closed.")
    parser.add_argument("--open-marker", default="eyes_open",
                        help="Marker name for eyes-open. Default eyes_open.")
    parser.add_argument("--closed-window", default=None,
                        help="Instead of a marker: 'start,end' in seconds.")
    parser.add_argument("--open-window", default=None,
                        help="Instead of a marker: 'start,end' in seconds.")
    parser.add_argument("--window", type=float, default=None,
                        help="Welch window length in seconds. Chosen automatically if omitted.")
    parser.add_argument("-o", "--output", default=None,
                        help="Where to write the figure. Default alongside the input.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the figure.")
    args = parser.parse_args(argv)

    recording = load(args.input, fs=args.fs, **({"gain": args.gain} if args.gain != 1.0 else {}))
    print(f"\nLoaded {args.input}")
    print(recording.summary())

    if not args.sites:
        print(
            f"\nPass --sites to choose the channels to analyse, for example "
            f"--sites {','.join(recording.channels[:2])}\n"
            f"Available: {', '.join(recording.channels)}",
            file=sys.stderr,
        )
        return 2

    results = analyse(
        recording,
        [s.strip() for s in args.sites.split(",") if s.strip()],
        alpha_band=parse_window(args.band),
        peak_search=parse_window(args.peak_search),
        closed_marker=args.closed_marker,
        open_marker=args.open_marker,
        closed_window=parse_window(args.closed_window),
        open_window=parse_window(args.open_window),
        window_seconds=args.window,
    )
    report(results)

    if not args.no_plot:
        source = Path(args.input)
        default = (source if source.is_dir() else source.parent) / "alpha_reactivity.png"
        written = plot(results, Path(args.output) if args.output else default)
        print(f"\n  Figure -> {written}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
