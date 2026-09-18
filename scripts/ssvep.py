#!/usr/bin/env python3
# Copyright 2026 Anthriq
# Licensed under the Apache License, Version 2.0. See the LICENSE file.
"""
Steady-state visually evoked potential (SSVEP): the response to a flickering
stimulus.

Look at something flickering at 12 Hz and the visual cortex produces a 12 Hz
rhythm, usually with a harmonic at 24 Hz. The response is narrow, strong, and
appears exactly where you put it, which makes it the basis of a large family of
brain-computer interfaces: show several targets flickering at different rates,
and the spectrum tells you which one the subject is looking at.

This script compares flicker against rest and reports:

  * band power and peak frequency at the stimulus frequency
  * signal-to-noise ratio at the fundamental and its harmonics
  * the phase-locking value, which is a different question from power

Usage
-----
    python scripts/ssvep.py my_recording/ --stim-freq 12 --sites O1,O2

    # With a frontal control channel, to show the response is occipital
    python scripts/ssvep.py my_recording/ --stim-freq 12 --sites O1,O2 --control Fpz

    # No hardware yet
    python scripts/synth.py ssvep /tmp/demo --stim-freq 12
    python scripts/ssvep.py /tmp/demo --stim-freq 12 --sites O1,O2 --control Fpz

Why --stim-freq is required
---------------------------
There is no default and no attempt to detect it. If the script searched for the
strongest peak and then reported its SNR, the answer would be guaranteed to look
good whether or not a response existed -- you would be measuring the largest
thing in the spectrum and calling it a response. The stimulus frequency is
something you know from how you set up the experiment, so you state it, and the
analysis then tests a claim that could fail.

Power versus phase locking
--------------------------
These answer different questions, and running both is what makes the result
trustworthy.

**Power** asks: is there more energy at 12 Hz during flicker than at rest? But
alpha sits at around 10 Hz and can extend to 12, so a subject with strong alpha
can show elevated 12 Hz power that has nothing to do with the stimulus.

**Phase locking** asks: does the 12 Hz activity start at the same phase on every
trial? A driven response does, because the stimulus resets it each time.
Spontaneous alpha does not. So high power with chance-level phase locking is the
signature of a rhythm you have not driven -- a distinction power alone cannot
make.

Note that the phase reference here is the stimulus marker, which is external to
the subject. Comparing the phase of two *electrodes* instead would be
confounded: both are recorded against a shared reference electrode, so whatever
that reference picks up appears in both signals and inflates any measure of
their similarity.

References
----------
Regan, D. (1966). Some characteristics of average steady-state and transient
    responses evoked by modulated light. Electroencephalography and Clinical
    Neurophysiology, 20(3), 238-248.
Norcia, A. M., Appelbaum, L. G., Ales, J. M., Cottereau, B. R., & Rossion, B.
    (2015). The steady-state visual evoked potential in vision research: a
    review. Journal of Vision, 15(6), 4.
Lachaux, J.-P., et al. (1999). Measuring phase synchrony in brain signals.
    Human Brain Mapping, 8(4), 194-208.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exg.io import Recording, load  # noqa: E402
from exg.plotting import EMBER, INK, MUTE, TEAL, save_figure, style_axes  # noqa: E402
from exg.spectra import (  # noqa: E402
    band_power,
    peak_in_band,
    phase_locking_value,
    rayleigh_threshold,
    snr_at_frequency,
    welch_psd,
)

# An SNR at or below this is not worth calling a response.
SNR_STRONG_DB = 6.0
SNR_USABLE_DB = 3.0


def condition_samples(
    recording: Recording,
    rows: np.ndarray,
    marker_name: str,
    *,
    window: tuple[float, float] | None = None,
) -> np.ndarray:
    """Concatenate every span belonging to one condition."""
    matches = [m for m in recording.markers_named(marker_name) if m.duration > 0]
    if matches:
        pieces = []
        for marker in matches:
            start = int(round(marker.onset * recording.fs))
            stop = int(round(marker.offset * recording.fs))
            piece = rows[:, max(0, start):min(rows.shape[1], stop)]
            if piece.shape[1] > 0:
                pieces.append(piece)
        if pieces:
            return np.concatenate(pieces, axis=1)

    if window is not None:
        i0 = int(round(window[0] * recording.fs))
        i1 = int(round(window[1] * recording.fs))
        return rows[:, max(0, i0):min(rows.shape[1], i1)]

    raise ValueError(
        f"No spans found for the '{marker_name}' condition. This recording's "
        f"markers are {recording.marker_names() or 'none'}."
    )


def build_epochs(
    recording: Recording,
    signal: np.ndarray,
    marker_name: str,
    epoch_seconds: float,
) -> np.ndarray:
    """Cut one epoch from each marker onset, for the phase-locking measurement.

    Every epoch starts at a stimulus onset, so sample 0 of each row is the
    instant the flicker began. That alignment is what makes the phase at sample
    0 comparable across trials.
    """
    onsets = [m.onset for m in recording.markers_named(marker_name)]
    if not onsets:
        raise ValueError(
            f"Phase locking needs stimulus onsets, but no '{marker_name}' markers "
            f"exist. Available: {recording.marker_names() or 'none'}. Without "
            "markers, power and SNR still work -- pass --no-plv."
        )

    length = int(round(epoch_seconds * recording.fs))
    epochs = []
    for onset in onsets:
        start = int(round(onset * recording.fs))
        stop = start + length
        if stop <= signal.size:
            epochs.append(signal[start:stop])

    if len(epochs) < 2:
        raise ValueError(
            f"Only {len(epochs)} complete epoch(s) of {epoch_seconds} s fit "
            "after the marker onsets. Phase locking needs at least 2; try a "
            "shorter --epoch."
        )
    return np.asarray(epochs)


def analyse(
    recording: Recording,
    sites: list[str],
    stim_freq: float,
    *,
    control: list[str] | None = None,
    n_harmonics: int = 3,
    stim_marker: str = "stimulus",
    rest_marker: str = "rest",
    stim_window: tuple[float, float] | None = None,
    rest_window: tuple[float, float] | None = None,
    epoch_seconds: float = 2.0,
    compute_plv: bool = True,
) -> dict:
    """Compare flicker against rest at a known stimulus frequency."""
    rows = recording.pick(sites)

    flicker = condition_samples(recording, rows, stim_marker, window=stim_window)
    rest = condition_samples(recording, rows, rest_marker, window=rest_window)

    # One window length for both conditions, long enough to resolve the peak.
    # A narrow response smeared across several bins loses amplitude, so err
    # toward longer windows here than you would for a broad rhythm.
    shortest = min(flicker.shape[1], rest.shape[1])
    nperseg = min(int(8 * recording.fs), shortest)

    freqs_f, psd_f = welch_psd(flicker.mean(axis=0), recording.fs, nperseg=nperseg)
    freqs_r, psd_r = welch_psd(rest.mean(axis=0), recording.fs, nperseg=nperseg)

    resolution = float(freqs_f[1] - freqs_f[0])

    # Harmonics: a flickering stimulus is not a pure sinusoid, so the response
    # appears at the driving frequency and at integer multiples of it. Checking
    # them matters because a genuine response usually shows at least one, while
    # narrowband interference typically does not.
    harmonics = []
    for k in range(1, n_harmonics + 1):
        frequency = stim_freq * k
        if frequency >= recording.fs / 2:
            break
        harmonics.append({
            "order": k,
            "frequency_hz": frequency,
            "snr_flicker_db": snr_at_frequency(freqs_f, psd_f, frequency),
            "snr_rest_db": snr_at_frequency(freqs_r, psd_r, frequency),
        })

    # Power in a narrow band around the fundamental, and where the peak actually
    # sits. If the peak is not within a bin or two of the stimulus frequency,
    # something is wrong with the stimulus timing rather than with the brain.
    half_width = max(0.5, 2 * resolution)
    power_flicker = float(band_power(freqs_f, psd_f, stim_freq - half_width, stim_freq + half_width))
    power_rest = float(band_power(freqs_r, psd_r, stim_freq - half_width, stim_freq + half_width))
    peak_freq, _ = peak_in_band(freqs_f, psd_f, stim_freq - 1.0, stim_freq + 1.0)

    results = {
        "sites": sites,
        "stim_freq_hz": stim_freq,
        "peak_frequency_hz": peak_freq,
        "peak_offset_hz": peak_freq - stim_freq,
        "power_flicker_uv2": power_flicker,
        "power_rest_uv2": power_rest,
        "power_ratio": power_flicker / power_rest if power_rest > 0 else float("inf"),
        "harmonics": harmonics,
        "frequency_resolution_hz": resolution,
        "flicker_seconds": flicker.shape[1] / recording.fs,
        "rest_seconds": rest.shape[1] / recording.fs,
        "_spectra": (freqs_f, psd_f, freqs_r, psd_r),
    }

    # Phase locking, on the site average.
    if compute_plv:
        epochs = build_epochs(recording, rows.mean(axis=0), stim_marker, epoch_seconds)
        results["plv"] = phase_locking_value(epochs, recording.fs, stim_freq)
        results["plv_n_epochs"] = int(epochs.shape[0])
        results["plv_chance_95"] = rayleigh_threshold(epochs.shape[0])
        results["_epochs"] = epochs

    # The control channel is what turns "there is a peak" into "there is an
    # occipital peak". A response that is just as large frontally is not a
    # visual evoked potential.
    if control:
        control_rows = recording.pick(control)
        control_flicker = condition_samples(recording, control_rows, stim_marker,
                                            window=stim_window)
        freqs_c, psd_c = welch_psd(control_flicker.mean(axis=0), recording.fs, nperseg=nperseg)
        results["control_sites"] = control
        results["control_snr_db"] = snr_at_frequency(freqs_c, psd_c, stim_freq)
        results["control_power_uv2"] = float(
            band_power(freqs_c, psd_c, stim_freq - half_width, stim_freq + half_width)
        )
        results["_control_spectrum"] = (freqs_c, psd_c)

    return results


def plot(results: dict, output: Path, *, plot_band: tuple[float, float] = (1.0, 45.0)) -> Path:
    """Spectra, SNR per harmonic, and the phase distribution."""
    import matplotlib.pyplot as plt

    freqs_f, psd_f, freqs_r, psd_r = results["_spectra"]
    stim = results["stim_freq_hz"]
    has_plv = "plv" in results

    n_panels = 3 if has_plv else 2
    figure = plt.figure(figsize=(4.7 * n_panels, 4.2))

    # Panel 1: spectra, flicker against rest.
    ax = figure.add_subplot(1, n_panels, 1)
    mask_f = (freqs_f >= plot_band[0]) & (freqs_f <= plot_band[1])
    mask_r = (freqs_r >= plot_band[0]) & (freqs_r <= plot_band[1])
    for harmonic in results["harmonics"]:
        ax.axvline(harmonic["frequency_hz"], color="0.85", lw=1.0, zorder=0)
    ax.semilogy(freqs_f[mask_f], psd_f[mask_f], color=EMBER, lw=1.5, label="flicker")
    ax.semilogy(freqs_r[mask_r], psd_r[mask_r], color=TEAL, lw=1.5, label="rest")
    if "_control_spectrum" in results:
        freqs_c, psd_c = results["_control_spectrum"]
        mask_c = (freqs_c >= plot_band[0]) & (freqs_c <= plot_band[1])
        ax.semilogy(freqs_c[mask_c], psd_c[mask_c], color=MUTE, lw=1.1, ls=":",
                    label=f"{', '.join(results['control_sites'])} (control)")
    ax.annotate(f"{stim:g} Hz", xy=(stim, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", fontsize=9, color=MUTE)
    style_axes(ax, xlabel="Frequency (Hz)", ylabel="PSD (uV$^2$/Hz)",
               title="Spectrum by condition")
    ax.legend(frameon=False, fontsize=8.5)

    # Panel 2: SNR per harmonic, flicker vs rest.
    ax = figure.add_subplot(1, n_panels, 2)
    labels = [f"{h['order']}f\n{h['frequency_hz']:g} Hz" for h in results["harmonics"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [h["snr_flicker_db"] for h in results["harmonics"]],
           width=0.4, color=EMBER, label="flicker")
    ax.bar(x + 0.2, [h["snr_rest_db"] for h in results["harmonics"]],
           width=0.4, color=TEAL, label="rest")
    ax.axhline(SNR_STRONG_DB, color=MUTE, ls="--", lw=0.9)
    ax.text(len(labels) - 0.5, SNR_STRONG_DB, f" {SNR_STRONG_DB:g} dB", fontsize=8,
            color=MUTE, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axes(ax, ylabel="SNR (dB)", title="Response at each harmonic")
    ax.legend(frameon=False, fontsize=8.5)

    # Panel 3: the phase of each trial as a unit vector, and their mean. The
    # length of the resultant IS the PLV, which makes the measure visible
    # rather than merely numerical.
    if has_plv:
        ax = figure.add_subplot(1, n_panels, 3, projection="polar")
        # One faint spoke per trial, pointing in the direction of that trial's
        # phase at stimulus onset, plus their resultant in bold. The resultant's
        # length is the PLV: aligned spokes give a long one, scattered spokes
        # cancel and give a short one.
        phases = np.asarray(results["_phases"])
        for phase in phases:
            ax.plot([phase, phase], [0, 1], color=EMBER, lw=1.0, alpha=0.45)
        resultant = np.mean(np.exp(1j * phases))
        ax.plot([np.angle(resultant), np.angle(resultant)], [0, np.abs(resultant)],
                color=INK, lw=2.6)
        circle = np.linspace(0, 2 * np.pi, 200)
        ax.plot(circle, np.full_like(circle, results["plv_chance_95"]),
                color=MUTE, ls="--", lw=1.0)
        ax.set_rlim(0, 1)
        ax.set_rticks([0.5, 1.0])
        ax.set_title(
            f"PLV {results['plv']:.3f}  (chance {results['plv_chance_95']:.2f})",
            fontsize=10.5, pad=14,
        )

    return save_figure(figure, output)


def report(results: dict) -> None:
    """Print the numbers with the context needed to read them."""
    print(f"\n  Sites               : {', '.join(results['sites'])}")
    print(f"  Stimulus frequency  : {results['stim_freq_hz']:g} Hz")
    print(f"  Frequency resolution: {results['frequency_resolution_hz']:.3f} Hz")
    print(f"  Data analysed       : {results['flicker_seconds']:.1f} s flicker, "
          f"{results['rest_seconds']:.1f} s rest")

    offset = results["peak_offset_hz"]
    print(f"\n  Peak found at       : {results['peak_frequency_hz']:.3f} Hz "
          f"({offset:+.3f} Hz from the stimulus)")
    if abs(offset) > max(0.25, 2 * results["frequency_resolution_hz"]):
        print("    The peak is not at the stimulus frequency. Check that the display")
        print("    really ran at the rate you asked for: a monitor can only flicker at")
        print("    its refresh rate divided by a whole number, so asking for 13 Hz on a")
        print("    60 Hz screen gives something between 12 and 15 Hz instead.")

    print(f"  Power at stimulus   : {results['power_flicker_uv2']:.2f} uV^2 flicker "
          f"vs {results['power_rest_uv2']:.2f} rest "
          f"({results['power_ratio']:.1f}x)")

    print("\n  Harmonic       Flicker      Rest")
    for harmonic in results["harmonics"]:
        print(f"  {harmonic['order']}f = {harmonic['frequency_hz']:6.2f} Hz "
              f"{harmonic['snr_flicker_db']:8.2f} dB {harmonic['snr_rest_db']:8.2f} dB")

    best = max((h["snr_flicker_db"] for h in results["harmonics"]), default=0.0)
    if best >= SNR_STRONG_DB:
        verdict = "a strong response"
    elif best >= SNR_USABLE_DB:
        verdict = "a usable but modest response"
    else:
        verdict = "no reliable response"
    print(f"\n  Best harmonic SNR   : {best:.2f} dB -> {verdict}")

    if "control_snr_db" in results:
        print(f"  Control {', '.join(results['control_sites']):<12}: "
              f"{results['control_snr_db']:.2f} dB at the stimulus frequency")
        if results["control_snr_db"] > best - 3.0:
            print("    The control channel responds nearly as strongly as the occipital")
            print("    sites, which is not what a visual evoked potential looks like.")
            print("    Suspect stimulus artefact reaching the electrodes, or a montage")
            print("    that is not where you think it is.")

    if "plv" in results:
        plv, chance = results["plv"], results["plv_chance_95"]
        print(f"\n  Phase locking       : {plv:.3f} over {results['plv_n_epochs']} trials "
              f"(chance {chance:.3f})")
        if plv > chance:
            print("    Above chance: the response is locked to the stimulus, so it is")
            print("    driven rather than spontaneous.")
        else:
            print("    At chance. If power at this frequency is nonetheless elevated, what")
            print("    you are seeing is most likely a spontaneous rhythm near the")
            print("    stimulus frequency -- alpha, if the stimulus is near 10 Hz -- rather")
            print("    than a driven response.")


def parse_pair(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    parts = text.replace(":", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'start,end', got {text!r}")
    return (float(parts[0]), float(parts[1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure a steady-state visually evoked potential.",
    )
    parser.add_argument("input", help="A BXI export folder, or a CSV file.")
    parser.add_argument("--stim-freq", type=float, required=True,
                        help="The flicker frequency in Hz. Required, and not guessed.")
    parser.add_argument("--sites", help="Recording channels, comma-separated, e.g. O1,O2.")
    parser.add_argument("--control", default=None,
                        help="Control channel(s), e.g. Fpz. Should respond far less.")
    parser.add_argument("--fs", type=float, default=None, help="Sample rate, for a bare CSV.")
    parser.add_argument("--gain", type=float, default=1.0, help="Amplifier gain for a raw capture.")
    parser.add_argument("--harmonics", type=int, default=3, help="How many harmonics to score.")
    parser.add_argument("--stim-marker", default="stimulus", help="Marker name for flicker blocks.")
    parser.add_argument("--rest-marker", default="rest", help="Marker name for rest blocks.")
    parser.add_argument("--stim-window", default=None, help="Instead of a marker: 'start,end' s.")
    parser.add_argument("--rest-window", default=None, help="Instead of a marker: 'start,end' s.")
    parser.add_argument("--epoch", type=float, default=2.0,
                        help="Epoch length for phase locking, in seconds. Default 2.")
    parser.add_argument("--no-plv", action="store_true", help="Skip phase locking.")
    parser.add_argument("-o", "--output", default=None, help="Where to write the figure.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the figure.")
    args = parser.parse_args(argv)

    recording = load(args.input, fs=args.fs,
                     **({"gain": args.gain} if args.gain != 1.0 else {}))
    print(f"\nLoaded {args.input}")
    print(recording.summary())

    if not args.sites:
        print(f"\nPass --sites, for example --sites {','.join(recording.channels[:2])}\n"
              f"Available: {', '.join(recording.channels)}", file=sys.stderr)
        return 2

    results = analyse(
        recording,
        [s.strip() for s in args.sites.split(",") if s.strip()],
        args.stim_freq,
        control=[s.strip() for s in args.control.split(",")] if args.control else None,
        n_harmonics=args.harmonics,
        stim_marker=args.stim_marker,
        rest_marker=args.rest_marker,
        stim_window=parse_pair(args.stim_window),
        rest_window=parse_pair(args.rest_window),
        epoch_seconds=args.epoch,
        compute_plv=not args.no_plv,
    )

    # Keep the per-trial phases for the polar panel.
    if "_epochs" in results:
        from scipy import signal as sp_signal
        sos = sp_signal.butter(4, [args.stim_freq - 1.0, args.stim_freq + 1.0],
                               btype="bandpass", fs=recording.fs, output="sos")
        filtered = sp_signal.sosfiltfilt(sos, results["_epochs"], axis=-1)
        results["_phases"] = np.angle(sp_signal.hilbert(filtered, axis=-1))[:, 0]

    report(results)

    if not args.no_plot:
        source = Path(args.input)
        default = (source if source.is_dir() else source.parent) / "ssvep_response.png"
        written = plot(results, Path(args.output) if args.output else default)
        print(f"\n  Figure -> {written}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
