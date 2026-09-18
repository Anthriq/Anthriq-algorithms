#!/usr/bin/env python3
# Copyright 2026 Nexstem India Private Limited (trading as Anthriq)
# Licensed under the Apache License, Version 2.0. See the LICENSE file.
"""
Surface electromyography (EMG): muscle activity through the skin.

Unlike EEG, surface EMG is large enough to see directly. A resting forearm sits
at tens of microvolts; a firm grip produces bursts of a millivolt or more. That
makes it the most satisfying first biosignal to record -- cause and effect are
visible in the raw trace, with no spectral analysis required.

This script reports:

  * an RMS envelope, which converts the raw burst into a smooth trace that
    tracks force
  * per-contraction amplitude, so the force-amplitude relationship is visible
  * median frequency over time, which is how muscle fatigue shows itself
  * the standard time-domain features used by gesture classifiers

Usage
-----
    python scripts/emg.py my_recording/ --sites EMG1,EMG2

    # A bare CSV: give the sample rate, and the gain if the file is
    # converter-referred rather than already in microvolts
    python scripts/emg.py grip.csv --fs 2000 --gain 2000 --sites ai0,ai1

    # No hardware yet
    python scripts/synth.py emg /tmp/demo && python scripts/emg.py /tmp/demo --sites EMG1

Recording a session worth analysing
-----------------------------------
A protocol that makes the force-amplitude relationship obvious:

1. **10 seconds of rest** to establish a baseline.
2. **Light grip, 5 s**, then 5 s rest.
3. **Medium grip, 5 s**, then 5 s rest.
4. **Maximum grip, 5 s**, then 5 s rest.

Mark the start of each contraction. For a fatigue run instead, hold about half
maximum for as long as the subject can manage, with a marker at the start and
at the point they report fatiguing.

Electrode placement matters more here than for EEG. Put the two recording
electrodes **along the muscle fibres, not across them**: a differential
amplifier sees the potential travelling along the fibre, so alignment directly
affects the amplitude you record. Put the reference on a bony landmark with
little muscle under it.

Why the envelope
----------------
Raw EMG is a burst of rapid oscillation with no consistent shape -- it is the
summed activity of many motor units firing asynchronously, so the waveform
itself is close to noise and its *mean* is near zero however hard the muscle
pulls. What carries the information is the amplitude of that noise, which is
why every EMG measure starts by rectifying or squaring.

The root-mean-square in a sliding window does this: square (so negative
excursions count), average over a window, take the root (to return to
microvolts). The window length is the one real choice -- long enough to smooth
the oscillation, short enough to follow a real change in force. 50-200 ms is
the usual range, and 100 ms is a reasonable default.

References
----------
De Luca, C. J. (1997). The use of surface electromyography in biomechanics.
    Journal of Applied Biomechanics, 13(2), 135-163.
Merletti, R., & Parker, P. A. (2004). Electromyography: Physiology, Engineering,
    and Noninvasive Applications. IEEE Press / Wiley.
Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature reduction
    and selection for EMG signal classification. Expert Systems with
    Applications, 39(8), 7420-7431.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exg.io import Recording, load  # noqa: E402
from exg.plotting import EMBER, INK, MUTE, TEAL, save_figure, style_axes  # noqa: E402
from exg.spectra import welch_psd  # noqa: E402

# Surface EMG's usable band. The low edge removes movement artefact and the
# slow baseline shift that comes with moving a limb; the high edge is where
# most amplifiers stop and where the signal-to-noise at the skin is poor
# anyway. Check what your own front end passes before widening it.
DEFAULT_BAND_HZ = (20.0, 300.0)

# Sliding window for the RMS envelope. Long enough to smooth the oscillation,
# short enough to follow a change in force.
DEFAULT_ENVELOPE_MS = 100.0


def bandpass(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    """Zero-phase band-pass, so burst onsets are not shifted in time.

    ``filtfilt`` runs the filter forwards and then backwards, cancelling the
    phase lag of the forward pass. That matters here: a one-directional filter
    delays the signal by a frequency-dependent amount, which would move the
    apparent onset of a contraction relative to its marker.
    """
    from scipy import signal as sp_signal

    nyquist = fs / 2.0
    high = min(high, nyquist * 0.95)
    if low >= high:
        raise ValueError(
            f"Band {low}-{high} Hz is empty at {fs} Hz sampling "
            f"(Nyquist is {nyquist:g} Hz). Record faster, or narrow the band."
        )
    sos = sp_signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, signal, axis=-1)


def rms_envelope(signal: np.ndarray, fs: float, window_ms: float) -> np.ndarray:
    """Root-mean-square amplitude in a sliding window.

    Computed as a moving average of the squared signal, then square-rooted. The
    moving average uses a cumulative sum, so the cost does not grow with the
    window length.
    """
    width = max(1, int(round(window_ms * 1e-3 * fs)))
    squared = np.asarray(signal, dtype=float) ** 2

    # Pad by half a window at each end so the envelope is the same length as
    # the signal and is not shifted relative to it.
    padded = np.pad(squared, (width // 2, width - width // 2 - 1), mode="edge")
    cumulative = np.cumsum(np.insert(padded, 0, 0.0))
    means = (cumulative[width:] - cumulative[:-width]) / width
    return np.sqrt(means[: signal.shape[-1]])


def median_frequency(signal: np.ndarray, fs: float) -> float:
    """The frequency below which half the signal's power lies.

    This is the standard fatigue index. As a muscle fatigues, the conduction
    velocity along its fibres falls, which stretches the motor-unit action
    potentials in time and therefore shifts their spectrum downward. The median
    frequency drops even while amplitude holds or rises, so the two together
    distinguish fatigue from simply pulling harder.
    """
    freqs, psd = welch_psd(signal, fs)
    band = (freqs >= 5.0) & (freqs <= fs / 2)
    freqs, psd = freqs[band], psd[band]
    cumulative = np.cumsum(psd)
    if cumulative[-1] <= 0:
        return float("nan")
    return float(freqs[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def time_domain_features(signal: np.ndarray, fs: float) -> dict:
    """The features a gesture classifier is usually built from.

    All three are cheap enough to compute in real time on a microcontroller,
    which is why they became standard rather than anything more sophisticated.
    """
    signal = np.asarray(signal, dtype=float)
    centred = signal - np.mean(signal)

    # Waveform length: the total path the trace travels. Captures amplitude and
    # frequency content together -- a bigger or busier signal both raise it.
    waveform_length = float(np.sum(np.abs(np.diff(signal))))

    # Zero crossings, with a deadband so that noise hovering around zero does
    # not register as hundreds of crossings. A rough frequency proxy.
    threshold = 0.01 * float(np.std(centred))
    crossings = int(np.sum(
        (centred[:-1] * centred[1:] < 0) & (np.abs(np.diff(centred)) > threshold)
    ))

    return {
        "rms_uv": float(np.sqrt(np.mean(signal ** 2))),
        "mean_absolute_uv": float(np.mean(np.abs(signal))),
        "waveform_length_uv": waveform_length,
        "zero_crossings_per_s": crossings / (len(signal) / fs),
        "median_frequency_hz": median_frequency(signal, fs),
    }


def find_bursts(
    envelope: np.ndarray,
    fs: float,
    *,
    baseline_seconds: float = 5.0,
    threshold_sd: float = 3.0,
    min_duration: float = 0.3,
) -> list[tuple[float, float]]:
    """Locate contractions as excursions of the envelope above rest.

    The threshold is set from the quietest part of the recording rather than
    from an absolute level in microvolts, because EMG amplitude varies hugely
    with electrode placement, skin condition and the individual. A threshold
    that works on one subject may never be crossed on another.
    """
    n_baseline = min(int(baseline_seconds * fs), len(envelope))
    quiet = envelope[:n_baseline] if n_baseline > 0 else envelope

    # Estimate the rest level and its spread robustly, from percentiles rather
    # than from the mean and standard deviation. Two reasons:
    #
    #  * a contraction accidentally caught in the baseline window would drag a
    #    mean upward and inflate a standard deviation, and
    #  * trimming the sample to avoid that (taking only its lower half, say)
    #    biases the spread downward instead, which puts the threshold *below*
    #    ordinary resting fluctuation and finds bursts everywhere.
    #
    # The median is unmoved by either problem, and the interquartile range
    # scaled by 1.349 estimates the standard deviation of the resting
    # distribution without being pulled by the tail. A floor of 5% of the
    # median keeps the threshold finite on a synthetically clean recording.
    median = float(np.median(quiet))
    iqr = float(np.percentile(quiet, 75) - np.percentile(quiet, 25))
    spread = max(iqr / 1.349, 0.05 * median)
    threshold = median + threshold_sd * spread

    above = envelope > threshold
    edges = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)

    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        stops.append(len(envelope))

    bursts = []
    for start, stop in zip(starts, stops):
        if (stop - start) / fs >= min_duration:
            bursts.append((start / fs, stop / fs))
    return bursts


def analyse(
    recording: Recording,
    sites: list[str],
    *,
    band: tuple[float, float] = DEFAULT_BAND_HZ,
    envelope_ms: float = DEFAULT_ENVELOPE_MS,
    baseline_seconds: float = 5.0,
    threshold_sd: float = 3.0,
) -> dict:
    """Measure contractions, their amplitudes, and fatigue indicators."""
    rows = recording.pick(sites)
    signal = rows.mean(axis=0) if rows.shape[0] > 1 else rows[0]

    filtered = bandpass(signal, recording.fs, *band)
    envelope = rms_envelope(filtered, recording.fs, envelope_ms)

    bursts = find_bursts(
        envelope, recording.fs,
        baseline_seconds=baseline_seconds, threshold_sd=threshold_sd,
    )

    # Rest is everything outside a burst, which is the fair baseline to compare
    # against -- taking the first N seconds assumes the subject was actually
    # still then.
    resting = np.ones(len(envelope), dtype=bool)
    for start, stop in bursts:
        resting[int(start * recording.fs):int(stop * recording.fs)] = False

    measured = []
    for index, (start, stop) in enumerate(bursts, start=1):
        segment = filtered[int(start * recording.fs):int(stop * recording.fs)]
        if segment.size < int(0.05 * recording.fs):
            continue
        features = time_domain_features(segment, recording.fs)
        features.update({
            "index": index,
            "onset_s": start,
            "duration_s": stop - start,
            "peak_envelope_uv": float(np.max(
                envelope[int(start * recording.fs):int(stop * recording.fs)]
            )),
        })
        measured.append(features)

    baseline_uv = float(np.median(envelope[resting])) if resting.any() else float("nan")

    return {
        "sites": sites,
        "band_hz": band,
        "envelope_ms": envelope_ms,
        "baseline_rms_uv": baseline_uv,
        "n_bursts": len(measured),
        "bursts": measured,
        "duration_s": recording.duration,
        "_signal": filtered,
        "_envelope": envelope,
        "_fs": recording.fs,
        "_markers": recording.markers,
    }


def plot(results: dict, output: Path) -> Path:
    """The trace with its envelope, burst amplitudes, and the fatigue view."""
    import matplotlib.pyplot as plt

    fs = results["_fs"]
    signal = results["_signal"]
    envelope = results["_envelope"]
    t = np.arange(len(signal)) / fs

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    # Panel 1: the raw band-passed trace with the envelope over it. This is the
    # panel that shows why the envelope is worth computing.
    ax = axes[0]
    ax.plot(t, signal, color=MUTE, lw=0.4, alpha=0.6, label="filtered EMG")
    ax.plot(t, envelope, color=EMBER, lw=1.6, label="RMS envelope")
    ax.plot(t, -envelope, color=EMBER, lw=1.6)
    for burst in results["bursts"]:
        ax.axvspan(burst["onset_s"], burst["onset_s"] + burst["duration_s"],
                   color=TEAL, alpha=0.12, zorder=0)
    style_axes(ax, xlabel="Time (s)", ylabel="Amplitude (uV)",
               title=f"{results['n_bursts']} contraction(s) detected")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Panel 2: amplitude per contraction, which is the force relationship.
    ax = axes[1]
    if results["bursts"]:
        indices = [b["index"] for b in results["bursts"]]
        peaks = [b["peak_envelope_uv"] for b in results["bursts"]]
        ax.bar(indices, peaks, color=EMBER, width=0.6)
        ax.axhline(results["baseline_rms_uv"], color=MUTE, ls="--", lw=1.0)
        ax.text(len(indices) + 0.4, results["baseline_rms_uv"], " rest",
                fontsize=8, color=MUTE, va="center", ha="right")
        ax.set_xticks(indices)
    style_axes(ax, xlabel="Contraction", ylabel="Peak envelope (uV)",
               title="Amplitude rises with force")

    # Panel 3: median frequency per contraction. Falling across a series of
    # equal-effort holds is the signature of fatigue.
    ax = axes[2]
    if results["bursts"]:
        indices = [b["index"] for b in results["bursts"]]
        medians = [b["median_frequency_hz"] for b in results["bursts"]]
        ax.plot(indices, medians, marker="o", ms=5, lw=1.6, color=TEAL)
        ax.set_xticks(indices)
    style_axes(ax, xlabel="Contraction", ylabel="Median frequency (Hz)",
               title="Falls as the muscle fatigues")

    return save_figure(figure, output)


def report(results: dict) -> None:
    """Print the measurements with the context needed to read them."""
    low, high = results["band_hz"]
    print(f"\n  Sites             : {', '.join(results['sites'])}")
    print(f"  Band              : {low:g}-{high:g} Hz")
    print(f"  Envelope window   : {results['envelope_ms']:g} ms")
    print(f"  Resting baseline  : {results['baseline_rms_uv']:.1f} uV RMS")
    print(f"  Contractions found: {results['n_bursts']}")

    if not results["bursts"]:
        print("\n  No contractions detected. If the subject was gripping, the usual")
        print("  causes are electrodes off the muscle belly, or a pair aligned across")
        print("  the fibres rather than along them. Check the raw trace before")
        print("  adjusting the detection threshold.")
        return

    print("\n   #   Onset   Dur    RMS      Peak    Median f   Waveform len")
    print("  " + "-" * 62)
    for burst in results["bursts"]:
        print(f"  {burst['index']:2d} {burst['onset_s']:7.2f}s {burst['duration_s']:5.2f}s "
              f"{burst['rms_uv']:8.1f} {burst['peak_envelope_uv']:8.1f} "
              f"{burst['median_frequency_hz']:8.1f}Hz {burst['waveform_length_uv']:12.0f}")

    peaks = [b["peak_envelope_uv"] for b in results["bursts"]]
    baseline = results["baseline_rms_uv"]
    if np.isfinite(baseline) and baseline > 0:
        print(f"\n  Strongest contraction is {max(peaks) / baseline:.0f}x the resting level.")

    if len(results["bursts"]) >= 3:
        medians = [b["median_frequency_hz"] for b in results["bursts"]]
        drop = medians[0] - medians[-1]
        if drop > 5.0:
            print(f"  Median frequency fell {drop:.1f} Hz from first contraction to last,")
            print("  which is the signature of fatigue: conduction velocity drops and the")
            print("  spectrum shifts down. Compare it against amplitude, which often holds")
            print("  or rises over the same period.")


def parse_pair(text: str) -> tuple[float, float]:
    parts = text.replace(":", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'low,high', got {text!r}")
    return (float(parts[0]), float(parts[1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure surface EMG contractions, amplitude and fatigue.",
    )
    parser.add_argument("input", help="A recording folder, or a CSV file.")
    parser.add_argument("--sites", help="Channels to analyse, comma-separated.")
    parser.add_argument("--fs", type=float, default=None,
                        help="Sample rate in Hz. Required for a bare CSV.")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="Amplifier gain, if the file is converter-referred.")
    parser.add_argument("--band", default="20,300",
                        help="Band-pass for viewing, in Hz. Default 20,300.")
    parser.add_argument("--envelope", type=float, default=DEFAULT_ENVELOPE_MS,
                        help="RMS window in milliseconds. Default 100.")
    parser.add_argument("--baseline", type=float, default=5.0,
                        help="Seconds at the start used to set the rest level. Default 5.")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Burst threshold, in SDs above rest. Default 3.")
    parser.add_argument("-o", "--output", default=None, help="Where to write the figure.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the figure.")
    args = parser.parse_args(argv)

    recording = load(args.input, fs=args.fs,
                     **({"gain": args.gain} if args.gain != 1.0 else {}))
    print(f"\nLoaded {args.input}")
    print(recording.summary())

    if not args.sites:
        print(f"\nPass --sites, for example --sites {recording.channels[0]}\n"
              f"Available: {', '.join(recording.channels)}", file=sys.stderr)
        return 2

    results = analyse(
        recording,
        [s.strip() for s in args.sites.split(",") if s.strip()],
        band=parse_pair(args.band),
        envelope_ms=args.envelope,
        baseline_seconds=args.baseline,
        threshold_sd=args.threshold,
    )
    report(results)

    if not args.no_plot:
        source = Path(args.input)
        default = (source if source.is_dir() else source.parent) / "emg_bursts.png"
        written = plot(results, Path(args.output) if args.output else default)
        print(f"\n  Figure -> {written}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
