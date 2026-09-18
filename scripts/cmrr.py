#!/usr/bin/env python3
# Copyright 2026 Nexstem India Private Limited (trading as Anthriq)
# Licensed under the Apache License, Version 2.0. See the LICENSE file.
"""
Common-mode rejection ratio (CMRR): how well an amplifier ignores what is
common to both its inputs.

Mains interference arrives on every electrode at once, at nearly the same
amplitude and phase. The biopotential you want appears as a *difference*
between electrodes. A differential amplifier is built to amplify the difference
and reject what is common, and CMRR states how well it manages that:

    CMRR (dB) = 20 * log10( common-mode signal / what leaks through )

A CMRR of 40 dB means an interfering signal is attenuated a hundredfold
relative to a differential one. This is the single number that decides whether
an EEG trace is readable in an ordinary room.

This is a bench measurement. No subject is involved, and nothing here touches a
person: you drive a known signal into the inputs through resistors and measure
what comes out.

Usage
-----
    # A directory of captures, one per drive frequency
    python scripts/cmrr.py captures/ --monitor ai0 --fs 2000

    # Name the channels under test explicitly
    python scripts/cmrr.py captures/ --monitor ai0 --channels ai1,ai2,ai3 --fs 2000

    # No bench yet
    python scripts/synth.py cmrr /tmp/demo_cmrr
    python scripts/cmrr.py /tmp/demo_cmrr --monitor Mod_9234/ai0 --fs 2000

Two things this script is built around
--------------------------------------
**CMRR is a curve, not a number.** A datasheet quotes one value, usually at DC
or at a low frequency, and it is natural to assume it holds everywhere. It does
not. Anything imperfectly matched between the amplifier's two input paths
contributes an error that varies with frequency, so the measurement slopes --
often by tens of decibels across the band you care about. Measure a sweep, and
if you must quote one number, say what frequency it was measured at.

**The measurement is designed so the gain cancels.** The obvious approach --
measure the differential gain, measure the common-mode gain, divide -- requires
knowing the amplifier's gain accurately, and any error in it propagates straight
into the answer. Instead, tie a *monitor* input into the same node that drives
the inputs under test, so the monitor sees the drive signal through the same
gain the channels do. The gain then appears in both the numerator and the
denominator of the ratio and cancels exactly:

    monitor reading  = G * V_drive
    channel reading  = G * V_drive / CMRR
    ratio            = CMRR          <- G is gone

So no calibration constant enters, and you do not need to know the gain at all.
You only need it to be the *same* for the monitor and the channels. A useful
consequence: running the same measurement at a different gain must produce the
same curve, which is a cheap way to check the rig is wired as you think.

References
----------
Winter, B. B., & Webster, J. G. (1983). Driven-right-leg circuit design. IEEE
    Transactions on Biomedical Engineering, 30(1), 62-66.
Metting van Rijn, A. C., Peper, A., & Grimbergen, C. A. (1990). High-quality
    recording of bioelectric events. Medical & Biological Engineering &
    Computing, 28(5), 389-397.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exg.io import load_daq_csv  # noqa: E402
from exg.plotting import EMBER, INK, MUTE, TEAL, save_figure, style_axes  # noqa: E402
from exg.spectra import tone_amplitude, welch_psd  # noqa: E402

# Pull a frequency out of a filename: "capture_50Hz.csv", "cap-12.5hz.csv",
# "sweep 100 Hz.csv" all work.
FREQUENCY_IN_NAME = re.compile(r"(?:^|[_\-\s])(\d+(?:\.\d+)?)\s*hz", re.IGNORECASE)

# A drive tone weaker than this above its own neighbourhood means the generator
# was probably off or disconnected for that capture.
MIN_DRIVE_SNR_DB = 10.0

# Mains frequencies, marked on the plot because CMRR there is the number that
# actually determines whether a recording is usable.
MAINS_FREQUENCIES = (50.0, 60.0)


def frequency_from_name(path: Path) -> float | None:
    """Extract the drive frequency from a capture's filename, or None."""
    match = FREQUENCY_IN_NAME.search(path.stem)
    return float(match.group(1)) if match else None


def drive_snr_db(signal: np.ndarray, fs: float, f0: float) -> float:
    """How far the tone at ``f0`` stands above the rest of the spectrum.

    Used only to decide whether a capture is usable. A capture recorded with the
    generator switched off contains no tone, and the ratio of two noise
    measurements is a meaningless CMRR that would otherwise join the sweep
    looking like data.
    """
    freqs, psd = welch_psd(signal, fs)
    target = int(np.argmin(np.abs(freqs - f0)))
    neighbourhood = np.concatenate([
        psd[max(0, target - 20):max(0, target - 2)],
        psd[min(len(psd), target + 3):min(len(psd), target + 21)],
    ])
    if neighbourhood.size == 0:
        return float("nan")
    return float(10.0 * np.log10(psd[target] / (np.mean(neighbourhood) + 1e-30) + 1e-30))


def measure_capture(
    path: Path,
    frequency: float,
    *,
    fs: float,
    monitor: str,
    channels: list[str] | None,
    settle_seconds: float,
) -> dict:
    """Measure CMRR for every channel in one capture, at one drive frequency."""
    # A CMRR rig drives volts into the inputs deliberately, so the loader's
    # "this does not look physiological" check would fire on every capture.
    recording = load_daq_csv(path, fs=fs, expect_physiological=False)

    # Drop the start of the capture: the generator's amplitude and the
    # amplifier's settling both take a moment, and including that transient
    # biases the amplitude estimate.
    skip = int(round(settle_seconds * fs))
    if skip >= recording.n_samples:
        raise ValueError(
            f"{path.name}: --settle {settle_seconds} s discards the whole "
            f"{recording.duration:.2f} s capture."
        )

    available = recording.channels
    monitor_row = recording.pick([monitor])[0, skip:]

    under_test = channels or [c for c in available if c != monitor]
    if not under_test:
        raise ValueError(
            f"{path.name}: no channels left to test after excluding the monitor "
            f"{monitor!r}. Columns present: {available}"
        )

    snr = drive_snr_db(monitor_row, fs, frequency)
    monitor_amplitude = tone_amplitude(monitor_row, fs, frequency)

    results = {
        "path": path,
        "frequency_hz": frequency,
        "monitor_uv": monitor_amplitude,
        "drive_snr_db": snr,
        "usable": bool(snr >= MIN_DRIVE_SNR_DB),
        "channels": {},
    }

    for name in under_test:
        row = recording.pick([name])[0, skip:]
        leak = tone_amplitude(row, fs, frequency)
        # The rejection ratio. Both amplitudes came through the same gain, so
        # the gain has already cancelled -- see the module docstring.
        cmrr = 20.0 * np.log10(monitor_amplitude / leak) if leak > 0 else float("inf")
        results["channels"][name] = {"leak_uv": leak, "cmrr_db": float(cmrr)}

    return results


def analyse(
    folder: Path,
    *,
    fs: float,
    monitor: str,
    channels: list[str] | None = None,
    settle_seconds: float = 0.5,
) -> dict:
    """Measure a CMRR sweep from a directory of captures."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(
            f"{folder} is not a directory. This measurement needs one capture "
            "per drive frequency, so point it at the folder holding them."
        )

    captures = sorted(p for p in folder.glob("*.csv") if p.is_file())
    if not captures:
        raise FileNotFoundError(f"No CSV captures in {folder}")

    measurements, skipped = [], []
    for path in captures:
        frequency = frequency_from_name(path)
        if frequency is None:
            skipped.append((path.name, "no frequency in the filename"))
            continue
        try:
            measurement = measure_capture(
                path, frequency, fs=fs, monitor=monitor,
                channels=channels, settle_seconds=settle_seconds,
            )
        except (ValueError, KeyError) as exc:
            skipped.append((path.name, str(exc)))
            continue

        if not measurement["usable"]:
            skipped.append((
                path.name,
                f"drive tone only {measurement['drive_snr_db']:.1f} dB above the "
                f"noise floor, below the {MIN_DRIVE_SNR_DB:g} dB threshold",
            ))
            continue
        measurements.append(measurement)

    if not measurements:
        raise ValueError(
            f"No usable captures in {folder}. Skipped: {skipped}"
        )

    measurements.sort(key=lambda m: m["frequency_hz"])
    names = list(measurements[0]["channels"])

    # Worst case across channels at each frequency. This is the figure a
    # datasheet should quote: a system is as good as its weakest channel.
    worst = [
        min(m["channels"][n]["cmrr_db"] for n in names if n in m["channels"])
        for m in measurements
    ]

    return {
        "monitor": monitor,
        "channel_names": names,
        "frequencies_hz": [m["frequency_hz"] for m in measurements],
        "measurements": measurements,
        "worst_case_db": worst,
        "skipped": skipped,
        "fs_hz": fs,
    }


def interpolate_at(frequencies: list[float], values: list[float], target: float) -> float | None:
    """Linear interpolation of the sweep at one frequency, on a log-f axis.

    Returns None outside the measured range. Extrapolating a CMRR curve is not
    safe: the slope changes, so a value beyond the last measured point would be
    a guess dressed as a measurement.
    """
    if not frequencies or target < min(frequencies) or target > max(frequencies):
        return None
    return float(np.interp(np.log10(target), np.log10(frequencies), values))


def plot(results: dict, output: Path) -> Path:
    """The sweep, and the two amplitudes it was computed from."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    frequencies = results["frequencies_hz"]

    # Panel 1: the result. Log frequency axis, because a sweep spans decades
    # and the interesting structure is at the low end.
    ax = axes[0]
    for mains in MAINS_FREQUENCIES:
        if min(frequencies) <= mains <= max(frequencies):
            ax.axvline(mains, color="0.88", lw=6.0, zorder=0)

    for name in results["channel_names"]:
        values = [
            m["channels"][name]["cmrr_db"]
            for m in results["measurements"] if name in m["channels"]
        ]
        ax.semilogx(frequencies, values, marker="o", ms=3.5, lw=1.0,
                    color=MUTE, alpha=0.65)
    ax.semilogx(frequencies, results["worst_case_db"], marker="o", ms=5,
                lw=2.2, color=EMBER, label="worst case", zorder=5)

    # Label the mains points once the data has set the axis limits, so the text
    # lands inside the frame.
    for mains in MAINS_FREQUENCIES:
        value = interpolate_at(frequencies, results["worst_case_db"], mains)
        if value is not None:
            ax.annotate(f"{value:.1f} dB\nat {mains:g} Hz", xy=(mains, value),
                        xytext=(-46, 6), textcoords="offset points",
                        fontsize=8.5, color=INK)

    style_axes(ax, xlabel="Drive frequency (Hz)", ylabel="CMRR (dB)",
               title=f"Rejection across frequency ({len(results['channel_names'])} channels)")
    ax.legend(frameon=False, fontsize=9)

    # Panel 2: the two measured amplitudes. This panel verifies the
    # measurement rather than reporting it -- you should see the rejection
    # curve rising because the leak falls, not because the drive drifts.
    ax = axes[1]
    ax.loglog(frequencies, [m["monitor_uv"] for m in results["measurements"]],
              marker="o", ms=4, lw=1.4, color=TEAL, label="monitor (common mode)")
    first = results["channel_names"][0]
    ax.loglog(
        frequencies,
        [m["channels"][first]["leak_uv"] for m in results["measurements"]],
        marker="s", ms=4, lw=1.4, color=EMBER, label=f"{first} (leak-through)",
    )
    style_axes(ax, xlabel="Drive frequency (Hz)", ylabel="Amplitude (uV)",
               title="What the ratio was computed from")
    ax.legend(frameon=False, fontsize=9)

    return save_figure(figure, output)


def report(results: dict) -> None:
    """Print the sweep as a table, plus the mains figures."""
    names = results["channel_names"]
    print(f"\n  Monitor channel : {results['monitor']}")
    print(f"  Under test      : {', '.join(names)}")
    print(f"  Sample rate     : {results['fs_hz']:g} Hz")

    header = "  Freq (Hz)  " + "".join(f"{n:>12s}" for n in names) + f"{'worst':>12s}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))
    for measurement, worst in zip(results["measurements"], results["worst_case_db"]):
        row = f"  {measurement['frequency_hz']:9.3f}  "
        for name in names:
            value = measurement["channels"].get(name, {}).get("cmrr_db", float("nan"))
            row += f"{value:12.2f}"
        row += f"{worst:12.2f}"
        print(row)

    print("\n  At mains frequency:")
    any_mains = False
    for mains in MAINS_FREQUENCIES:
        value = interpolate_at(results["frequencies_hz"], results["worst_case_db"], mains)
        if value is None:
            continue
        any_mains = True
        measured = mains in results["frequencies_hz"]
        how = "measured" if measured else "interpolated between measured points"
        print(f"    {mains:g} Hz : {value:.2f} dB   ({how})")
    if not any_mains:
        print(f"    Not available: the sweep covers "
              f"{min(results['frequencies_hz']):g}-{max(results['frequencies_hz']):g} Hz, "
              "which does not reach a mains frequency. Extend the sweep rather than")
        print("    extrapolating, since the slope changes across the band.")

    if results["skipped"]:
        print(f"\n  Skipped {len(results['skipped'])} capture(s):")
        for name, reason in results["skipped"]:
            print(f"    {name}: {reason}")

    print("\n  Report a CMRR figure with the conditions that produced it: the drive")
    print("  frequency, the drive amplitude, the source impedances, and the DAQ and its")
    print("  input range. The same amplifier measures differently on a different source,")
    print("  so a number quoted without them cannot be reproduced or compared.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure common-mode rejection across frequency from a set of captures.",
    )
    parser.add_argument("input", help="Directory of captures, one CSV per drive frequency.")
    parser.add_argument("--monitor", required=True,
                        help="Column carrying the common-mode drive, e.g. ai0. Required: it "
                             "is the reference the ratio is taken against.")
    parser.add_argument("--fs", type=float, required=True,
                        help="Sample rate in Hz. Required, since a raw capture carries no timing.")
    parser.add_argument("--channels", default=None,
                        help="Channels under test, comma-separated. Default: all but the monitor.")
    parser.add_argument("--settle", type=float, default=0.5,
                        help="Seconds to discard from the start of each capture. Default 0.5.")
    parser.add_argument("-o", "--output", default=None, help="Where to write the figure.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the figure.")
    args = parser.parse_args(argv)

    results = analyse(
        Path(args.input),
        fs=args.fs,
        monitor=args.monitor,
        channels=[c.strip() for c in args.channels.split(",")] if args.channels else None,
        settle_seconds=args.settle,
    )
    report(results)

    if not args.no_plot:
        default = Path(args.input) / "cmrr_vs_frequency.png"
        written = plot(results, Path(args.output) if args.output else default)
        print(f"\n  Figure -> {written}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
