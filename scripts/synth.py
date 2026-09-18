#!/usr/bin/env python3
# Copyright 2026 Nexstem India Private Limited (trading as Anthriq)
# Licensed under the Apache License, Version 2.0. See the LICENSE file.
"""
Generate synthetic recordings with known answers.

Two reasons this exists, and the second is the more important one.

**You can run everything before the hardware arrives.** Every analysis in this
repository works on a generated dataset, so you can learn the tools, read the
code and see a finished figure on day one.

**You know what the answer should be.** Real data never tells you whether your
analysis is right -- if the number looks plausible, you believe it. Synthetic
data has a ground truth you chose, so you can check that the analysis recovers
it. That is the only way to tell a working pipeline from a broken one, and it is
worth doing before you trust any tool, including this one.

The files written are in the *real* input layouts, so they exercise the same
readers your own recordings will go through.

Usage
-----
    python scripts/synth.py alpha /tmp/demo_alpha
    python scripts/synth.py ssvep /tmp/demo_ssvep --stim-freq 12
    python scripts/synth.py cmrr  /tmp/demo_cmrr

Each prints the ground truth it used, so you can compare it with what the
analysis reports.

What is being modelled
----------------------
A real EEG spectrum falls off roughly as 1/f: there is far more power at low
frequencies than high. That background ("aperiodic" or "pink" noise) is what
makes a rhythm hard to see, so the generator reproduces it rather than using
white noise, which would make every analysis look better than it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pink_noise(n: int, fs: float, rng: np.random.Generator, *, exponent: float = 1.0) -> np.ndarray:
    """Generate noise whose power spectrum falls as 1/f**exponent.

    Made by shaping white noise in the frequency domain: take the FFT of white
    noise, divide each bin by f**(exponent/2) -- half the exponent, because
    power is amplitude squared -- and transform back.
    """
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    scale = np.ones_like(freqs)
    nonzero = freqs > 0
    scale[nonzero] = freqs[nonzero] ** (-exponent / 2.0)
    scale[~nonzero] = 0.0  # drop DC; an offset is not part of the noise model

    shaped = np.fft.irfft(spectrum * scale, n=n)
    std = np.std(shaped)
    return shaped / std if std > 0 else shaped


def write_bxi_export(
    folder: Path,
    data_uv: np.ndarray,
    fs: float,
    channels: list[str],
    markers: list[tuple[str, float, float]],
) -> Path:
    """Write a folder in BXI Studio export layout.

    Encodes the markers in the same dictionary-plus-delta form BXI uses, so the
    reader's decoder is tested against an independent encoder rather than
    against a fixture that might share its bugs.
    """
    folder = Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    name = folder.name

    n_samples = data_uv.shape[1]

    # Timestamps: absolute microseconds, as BXI writes them.
    base_us = 1_718_000_000_000_000
    timestamps = base_us + np.round(np.arange(n_samples) * 1e6 / fs).astype(np.int64)

    csv_path = folder / f"{name}.csv"
    header = ",".join(channels) + ",timestamp"
    table = np.column_stack([data_uv.T, timestamps])
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%.6f"] * len(channels) + ["%d"],
    )

    # Build the marker block. Distinct names become definitions; occurrences
    # become parallel arrays with dt as the gap from the previous one.
    ordered = sorted(markers, key=lambda m: m[1])
    names = list(dict.fromkeys(m[0] for m in ordered))
    definitions = [
        {
            "id": index,
            "name": marker_name,
            "kind": "epoch" if any(m[0] == marker_name and m[2] > 0 for m in ordered) else "event",
            "source": "experiment",
        }
        for index, marker_name in enumerate(names)
    ]
    name_to_id = {marker_name: index for index, marker_name in enumerate(names)}

    def_ids, deltas, durations = [], [], []
    previous_us = 0
    for marker_name, onset, duration in ordered:
        onset_us = int(round(onset * 1e6))
        def_ids.append(name_to_id[marker_name])
        deltas.append(onset_us - previous_us)  # the gap, not the absolute time
        durations.append(int(round(duration * 1e6)))
        previous_us = onset_us

    meta = {
        "recordingId": name,
        "deviceName": "synthetic",
        "sampleRate": float(fs),
        "numChannels": len(channels),
        "channels": [{"id": f"c{i}", "label": c, "unit": "uV"} for i, c in enumerate(channels)],
        "totalSamples": n_samples,
        "kind": "experiment",
    }
    if ordered:
        meta["markers"] = {
            "v": 1,
            "defs": definitions,
            "seq": {"def": def_ids, "dt": deltas, "dur": durations, "data": {}},
            "timeBase": "us",
            "origin": "recordingStart",
            "count": len(ordered),
        }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return folder


def make_alpha(
    out: Path,
    *,
    fs: float = 250.0,
    n_blocks: int = 3,
    closed_seconds: float = 30.0,
    open_seconds: float = 15.0,
    alpha_freq: float = 10.2,
    amp_closed: float = 20.0,
    amp_open: float = 5.0,
    noise_uv: float = 8.0,
    seed: int = 0,
) -> tuple[Path, dict]:
    """Alternating eyes-closed and eyes-open blocks with a known alpha rhythm.

    Channels are named O1, O2 and Fpz: the two occipital sites carry the rhythm,
    the frontal one gets only a tenth of it, mimicking the real spatial pattern
    that makes Fpz a useful control.
    """
    rng = np.random.default_rng(seed)
    channels = ["O1", "O2", "Fpz"]
    occipital_weights = [1.0, 0.8, 0.1]

    block = []
    markers: list[tuple[str, float, float]] = []
    cursor = 0.0
    for _ in range(n_blocks):
        for label, seconds, amplitude in (
            ("eyes_closed", closed_seconds, amp_closed),
            ("eyes_open", open_seconds, amp_open),
        ):
            n = int(round(seconds * fs))
            t = np.arange(n) / fs
            # A random starting phase per block: a real rhythm is not
            # phase-locked to the instruction to close your eyes.
            phase = rng.uniform(0, 2 * np.pi)
            rhythm = np.sin(2 * np.pi * alpha_freq * t + phase)
            block.append((label, n, amplitude, rhythm))
            markers.append((label, cursor, seconds))
            cursor += seconds

    total = sum(n for _, n, _, _ in block)
    data = np.zeros((len(channels), total))
    for ch, weight in enumerate(occipital_weights):
        offset = 0
        for _, n, amplitude, rhythm in block:
            data[ch, offset:offset + n] = weight * amplitude * rhythm
            offset += n
        # The 1/f background, independent per channel.
        data[ch] += noise_uv * pink_noise(total, fs, rng)

    folder = write_bxi_export(out, data, fs, channels, markers)

    # The truth, for comparison with what the analysis reports. A sinusoid of
    # peak amplitude A has mean-square power A**2/2, so that is the band power
    # the analysis should recover (plus whatever the noise contributes in-band).
    truth = {
        "alpha_freq_hz": alpha_freq,
        "expected_closed_uv2_signal_only": amp_closed ** 2 / 2,
        "expected_open_uv2_signal_only": amp_open ** 2 / 2,
        "expected_ratio_signal_only": (
            (amp_closed / amp_open) ** 2 if amp_open > 0 else float("inf")
        ),
        "occipital_sites": "O1,O2",
        "control_site": "Fpz",
    }
    return folder, truth


def make_ssvep(
    out: Path,
    *,
    fs: float = 500.0,
    stim_freq: float = 12.0,
    n_trials: int = 12,
    flicker_seconds: float = 10.0,
    rest_seconds: float = 5.0,
    amp_fundamental: float = 8.0,
    amp_harmonic: float = 3.0,
    phase_jitter: float = 0.0,
    noise_uv: float = 8.0,
    seed: int = 0,
) -> tuple[Path, dict]:
    """Flicker and rest blocks with a phase-locked response at ``stim_freq``.

    With ``phase_jitter=0`` the response starts at the same phase on every
    trial, so its phase-locking value should come out near 1. Raising the jitter
    toward 2*pi scatters the phases and drives PLV toward chance -- which is how
    you check that a PLV you measured means anything.
    """
    rng = np.random.default_rng(seed)
    channels = ["O1", "O2", "Fpz"]
    weights = [1.0, 0.85, 0.15]

    segments = []
    markers: list[tuple[str, float, float]] = []
    cursor = 0.0
    for _ in range(n_trials):
        n = int(round(flicker_seconds * fs))
        t = np.arange(n) / fs
        phase = rng.uniform(-phase_jitter / 2, phase_jitter / 2) if phase_jitter else 0.0
        response = (
            amp_fundamental * np.sin(2 * np.pi * stim_freq * t + phase)
            + amp_harmonic * np.sin(2 * np.pi * 2 * stim_freq * t + phase)
        )
        segments.append((n, response))
        markers.append(("stimulus", cursor, flicker_seconds))
        cursor += flicker_seconds

        n_rest = int(round(rest_seconds * fs))
        segments.append((n_rest, np.zeros(n_rest)))
        markers.append(("rest", cursor, rest_seconds))
        cursor += rest_seconds

    total = sum(n for n, _ in segments)
    data = np.zeros((len(channels), total))
    for ch, weight in enumerate(weights):
        offset = 0
        for n, response in segments:
            data[ch, offset:offset + n] = weight * response
            offset += n
        data[ch] += noise_uv * pink_noise(total, fs, rng)

    folder = write_bxi_export(out, data, fs, channels, markers)

    truth = {
        "stim_freq_hz": stim_freq,
        "expected_peak_hz": stim_freq,
        "expected_harmonic_hz": 2 * stim_freq,
        "expected_plv": "~1.0" if phase_jitter == 0 else f"reduced (jitter {phase_jitter:.2f} rad)",
        "n_trials": n_trials,
        "occipital_sites": "O1,O2",
        "control_site": "Fpz",
    }
    return folder, truth


def make_cmrr(
    out: Path,
    *,
    fs: float = 2000.0,
    frequencies: tuple[float, ...] = (1.5, 3.0, 6.0, 12.0, 25.0, 50.0, 100.0),
    gain: float = 100.0,
    drive_v: float = 0.1,
    capture_seconds: float = 4.0,
    n_channels: int = 3,
    seed: int = 0,
) -> tuple[Path, dict]:
    """A frequency sweep of captures for a common-mode rejection measurement.

    One CSV per drive frequency, in raw DAQ layout. The rejection is modelled as
    rising with frequency, which is what a real front end limited by a
    high-pass corner mismatch between its two legs actually does -- the opposite
    of the usual intuition that rejection gets worse as frequency rises.

    Both the monitor and the channels under test are multiplied by the same
    ``gain``, so the analysis must return the same rejection curve whatever gain
    is used. That invariance is the point of the node-referenced method, and it
    is worth verifying by generating two sweeps with different gains.
    """
    rng = np.random.default_rng(seed)
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    def rejection_db(f: float) -> float:
        # ~13 dB at 1.5 Hz rising about 6 dB per octave, flattening at the top.
        return float(13.0 + 20.0 * np.log10(max(f, 0.1) / 1.5) * 0.85)

    truth_curve = {}
    n = int(round(capture_seconds * fs))
    t = np.arange(n) / fs

    for freq in frequencies:
        cmrr_db = rejection_db(freq)
        # The common-mode signal, as seen on the monitor channel.
        monitor = drive_v * gain * np.sin(2 * np.pi * freq * t)
        # What leaks through differentially, smaller by the rejection ratio.
        leak_amplitude = drive_v * gain / (10 ** (cmrr_db / 20.0))

        columns = [monitor + 1e-6 * gain * rng.standard_normal(n)]
        for _ in range(n_channels):
            columns.append(
                leak_amplitude * np.sin(2 * np.pi * freq * t + rng.uniform(0, 0.2))
                + 1e-6 * gain * rng.standard_normal(n)
            )

        header = "Mod_9234/ai0," + ",".join(f"Mod_9234/ai{i + 1}" for i in range(n_channels))
        np.savetxt(
            out / f"capture_{freq:g}Hz.csv",
            np.column_stack(columns),
            delimiter=",",
            header=header,
            comments="",
            fmt="%.9f",
        )
        truth_curve[freq] = round(cmrr_db, 2)

    truth = {
        "monitor_channel": "Mod_9234/ai0",
        "test_channels": [f"Mod_9234/ai{i + 1}" for i in range(n_channels)],
        "fs_hz": fs,
        "gain": gain,
        "expected_cmrr_db": truth_curve,
        "note": "The same curve must come out at any gain; that is the method's point.",
    }
    return out, truth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic recordings with known ground truth.",
    )
    sub = parser.add_subparsers(dest="kind", required=True)

    p_alpha = sub.add_parser("alpha", help="Eyes-closed / eyes-open blocks.")
    p_alpha.add_argument("output")
    p_alpha.add_argument("--fs", type=float, default=250.0)
    p_alpha.add_argument("--alpha-freq", type=float, default=10.2)
    p_alpha.add_argument("--amp-closed", type=float, default=20.0)
    p_alpha.add_argument("--amp-open", type=float, default=5.0)
    p_alpha.add_argument("--noise", type=float, default=8.0)
    p_alpha.add_argument("--seed", type=int, default=0)

    p_ssvep = sub.add_parser("ssvep", help="Flicker / rest trials.")
    p_ssvep.add_argument("output")
    p_ssvep.add_argument("--fs", type=float, default=500.0)
    p_ssvep.add_argument("--stim-freq", type=float, default=12.0)
    p_ssvep.add_argument("--trials", type=int, default=12)
    p_ssvep.add_argument("--phase-jitter", type=float, default=0.0,
                         help="Radians of trial-to-trial phase scatter. 0 gives PLV ~1; "
                              "6.28 gives chance.")
    p_ssvep.add_argument("--noise", type=float, default=8.0)
    p_ssvep.add_argument("--seed", type=int, default=0)

    p_cmrr = sub.add_parser("cmrr", help="A common-mode rejection frequency sweep.")
    p_cmrr.add_argument("output")
    p_cmrr.add_argument("--fs", type=float, default=2000.0)
    p_cmrr.add_argument("--gain", type=float, default=100.0)
    p_cmrr.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    if args.kind == "alpha":
        path, truth = make_alpha(
            Path(args.output), fs=args.fs, alpha_freq=args.alpha_freq,
            amp_closed=args.amp_closed, amp_open=args.amp_open,
            noise_uv=args.noise, seed=args.seed,
        )
    elif args.kind == "ssvep":
        path, truth = make_ssvep(
            Path(args.output), fs=args.fs, stim_freq=args.stim_freq,
            n_trials=args.trials, phase_jitter=args.phase_jitter,
            noise_uv=args.noise, seed=args.seed,
        )
    else:
        path, truth = make_cmrr(Path(args.output), fs=args.fs, gain=args.gain, seed=args.seed)

    print(f"\nWrote a synthetic {args.kind} dataset to {path}")
    print("\nGround truth (compare this with what the analysis reports):")
    for key, value in truth.items():
        print(f"  {key:34s} {value}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
