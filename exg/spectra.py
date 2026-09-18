# Copyright 2026 Nexstem India Private Limited (trading as Anthriq)
# Licensed under the Apache License, Version 2.0. See the LICENSE file.

"""
Spectral estimation for biosignals: PSD, band power, peaks, SNR, phase locking.

Everything here is plain NumPy and SciPy. Read it — the functions are short, and
the comments explain the signal processing rather than the syntax.

--------------------------------------------------------------------------------
UNITS — the one contract worth memorising
--------------------------------------------------------------------------------
Pass signals in **microvolts (uV)**. Then:

    welch_psd   -> uV**2/Hz      (power spectral density)
    band_power  -> uV**2         (density integrated over a frequency range)
    tone_amplitude -> uV         (peak amplitude of a sinusoid)

Nothing in this module rescales your data. If you hand it volts, you get
volts-squared back, and every number will look 1e12 too small. ``exg.io``
converts to microvolts on load precisely so this stays simple.

--------------------------------------------------------------------------------
A note for future maintainers
--------------------------------------------------------------------------------
This module is the single place where the spectral backend lives. If a public
core library is ever released with these same primitives, the function bodies
here can delegate to it while the signatures stay put, so no analysis script and
no student's fork has to change. Keep that boundary intact: scripts call these
functions, never SciPy directly.

References
----------
Welch, P. D. (1967). The use of Fast Fourier Transform for the estimation of
    power spectra. IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.
Lachaux, J.-P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999). Measuring
    phase synchrony in brain signals. Human Brain Mapping, 8(4), 194-208.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal as sp_signal

__all__ = [
    "choose_nperseg",
    "welch_psd",
    "band_power",
    "peak_in_band",
    "snr_at_frequency",
    "tone_amplitude",
    "phase_locking_value",
    "rayleigh_threshold",
]


# --------------------------------------------------------------------------
# Choosing the window length
# --------------------------------------------------------------------------

def choose_nperseg(
    n_samples: int,
    fs: float,
    *,
    target_segments: int = 8,
    min_seconds: float = 1.0,
    max_seconds: float = 8.0,
) -> int:
    """Pick a Welch segment length (in samples) from the recording length.

    Why this is a function and not the number 2048
    ----------------------------------------------
    ``nperseg`` is a *sample count*, so the same literal means different things
    at different sampling rates. 2048 samples is 8.2 s at 250 Hz but only 0.5 s
    at 4 kHz -- and 0.5 s gives 2 Hz frequency bins, which is wider than the
    entire alpha band is interesting. Hard-coding it is a real and costly
    mistake: a sweep on this very point found roughly 14-16 dB of steady-state
    response being thrown away purely because the window was too short.

    The trade-off
    -------------
    A longer window buys finer frequency resolution (bin spacing is ``fs/nperseg``
    Hz) but leaves fewer segments to average, so the estimate gets noisier. Welch
    averages periodograms, and the variance falls roughly as 1/(number of
    segments). Aiming for about 8 segments is a reasonable middle: enough
    averaging to be stable, long enough windows to resolve a narrow peak.

    With 50% overlap, ``n_segments ~= 2*n_samples/nperseg - 1``, so solving for
    the target gives ``nperseg = 2*n_samples/(target_segments + 1)``.

    Parameters
    ----------
    n_samples : int
        Length of the signal you are about to analyse.
    fs : float
        Sampling rate in Hz.
    target_segments : int
        How many overlapping segments to aim for. Default 8.
    min_seconds, max_seconds : float
        Clamp the window to this range. The floor keeps frequency resolution
        usable (1 s gives 1 Hz bins); the ceiling stops a long recording
        producing a window so long that nothing is averaged.

    Returns
    -------
    int
        Segment length in samples, never longer than the signal itself.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if fs <= 0:
        raise ValueError("fs must be positive")

    ideal = 2 * n_samples / (target_segments + 1)
    lo, hi = min_seconds * fs, max_seconds * fs
    nperseg = int(round(min(max(ideal, lo), hi)))

    # Never ask for more samples than we have, and keep a sane floor so very
    # short recordings still produce something rather than raising.
    return max(64, min(nperseg, n_samples))


# --------------------------------------------------------------------------
# Power spectral density
# --------------------------------------------------------------------------

def welch_psd(
    x: np.ndarray,
    fs: float,
    *,
    nperseg: int | None = None,
    overlap: float = 0.5,
    window: str = "hann",
    detrend: str = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density by Welch's method.

    Welch's method cuts the signal into overlapping segments, windows each one,
    takes its periodogram, and averages them. The averaging is the point: a
    single FFT of a noisy signal is itself extremely noisy, and averaging 8
    segments cuts the standard error by roughly sqrt(8).

    Choices made here, and why
    --------------------------
    * **Hann window.** Every segment is multiplied by a taper that goes to zero
      at both ends. Without it, the abrupt cut at each segment boundary acts
      like a step change and smears energy across the whole spectrum ("spectral
      leakage"). Hann's sidelobes fall away far faster than a rectangular
      window's, which matters when you are looking for a small peak sitting near
      a large one -- an evoked response next to mains hum, say.
    * **50% overlap.** With a Hann taper the segment edges are attenuated, so
      non-overlapping segments effectively throw those samples away. Overlapping
      by half recovers them at almost no cost in independence.
    * **detrend="constant".** Subtract each segment's mean, removing the DC
      offset that would otherwise dominate the lowest bins. Deliberately *not*
      "linear": a linear detrend would also remove genuine slow drift, which is
      information about electrode quality you want to be able to see.
    * **scaling="density"** (SciPy's default). Output is power per hertz, so the
      value at a bin does not change when you change the window length. That is
      what makes band power (an integral over Hz) comparable between recordings
      analysed with different windows.

    Parameters
    ----------
    x : ndarray
        Signal in microvolts. 1-D, or 2-D as (n_channels, n_samples).
    fs : float
        Sampling rate in Hz.
    nperseg : int, optional
        Segment length in samples. Chosen by :func:`choose_nperseg` if omitted.
    overlap : float
        Fraction of a segment to overlap, 0 to <1. Default 0.5.
    window : str
        Any SciPy window name. Default "hann".
    detrend : str
        Passed to SciPy. Default "constant".

    Returns
    -------
    freqs : ndarray
        Bin centre frequencies in Hz. Spacing is ``fs/nperseg``.
    psd : ndarray
        Power spectral density in uV**2/Hz, same leading shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    n_samples = x.shape[-1]
    if nperseg is None:
        nperseg = choose_nperseg(n_samples, fs)
    nperseg = int(min(nperseg, n_samples))

    freqs, psd = sp_signal.welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=int(overlap * nperseg),
        detrend=detrend,
        scaling="density",
        axis=-1,
    )
    return freqs, psd


# --------------------------------------------------------------------------
# Integrating a band
# --------------------------------------------------------------------------

def band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
) -> float | np.ndarray:
    """Integrate a PSD over [fmin, fmax], returning power in uV**2.

    Band power is the area under the density curve across the band, so this is
    a numerical integral -- ``np.trapezoid`` over the bins that fall inside it.

    The single-bin trap
    -------------------
    Trapezoidal integration over *one* point returns exactly **zero**, because a
    trapezoid needs two edges to have any area. If the requested band is
    narrower than the bin spacing, a naive trapezoid silently reports 0.000 --
    which is indistinguishable from "we measured no power here". This has caused
    real confusion, so a single bin is instead integrated as
    ``density * bin_width``, which is the right answer and never silently zero.

    Parameters
    ----------
    freqs : ndarray
        Bin frequencies from :func:`welch_psd`.
    psd : ndarray
        Density in uV**2/Hz. 1-D, or 2-D as (n_channels, n_freqs).
    fmin, fmax : float
        Band edges in Hz. ``fmin`` inclusive, ``fmax`` inclusive.

    Returns
    -------
    float or ndarray
        Power in uV**2. Scalar for 1-D input, one value per channel for 2-D.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)

    if fmax <= fmin:
        raise ValueError(f"fmax ({fmax}) must exceed fmin ({fmin})")

    mask = (freqs >= fmin) & (freqs <= fmax)
    n_bins = int(mask.sum())

    if n_bins == 0:
        # The band falls between bins entirely. Say so rather than returning 0.
        bin_width = float(freqs[1] - freqs[0]) if freqs.size > 1 else float("nan")
        raise ValueError(
            f"No PSD bins fall in {fmin}-{fmax} Hz (bin spacing is {bin_width:.4g} Hz). "
            "Use a longer analysis window for finer frequency resolution."
        )

    if n_bins == 1:
        # See "The single-bin trap" above: density * width, not a trapezoid.
        bin_width = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
        return psd[..., mask].squeeze(-1) * bin_width

    return np.trapezoid(psd[..., mask], x=freqs[mask], axis=-1)


# --------------------------------------------------------------------------
# Finding a peak
# --------------------------------------------------------------------------

def peak_in_band(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
    *,
    interpolate: bool = True,
) -> tuple[float, float]:
    """Locate the largest spectral peak within a band.

    Returns the peak frequency and its density. With ``interpolate=True`` the
    frequency is refined below the bin spacing by fitting a parabola through the
    largest bin and its two neighbours, in log power.

    Why interpolate
    ---------------
    A real rhythm almost never sits exactly on a bin centre, so the bin with the
    most power is only within half a bin of the truth. At 0.25 Hz spacing that
    is +/-0.125 Hz, which is enough to blur the distinction between two people's
    individual alpha frequencies. Three-point parabolic interpolation is the
    standard cheap fix and typically gets within a few hundredths of a hertz.
    Fitting in log power rather than linear power matters because a spectral
    peak is closer to Gaussian on a log scale, and a parabola fits a Gaussian's
    log exactly.

    Parameters
    ----------
    freqs, psd : ndarray
        From :func:`welch_psd`. ``psd`` must be 1-D (one channel).
    fmin, fmax : float
        Search range in Hz.
    interpolate : bool
        Refine the peak below bin spacing. Default True.

    Returns
    -------
    peak_freq : float
        Frequency of the peak in Hz.
    peak_psd : float
        Density at the peak, in uV**2/Hz.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if psd.ndim != 1:
        raise ValueError("peak_in_band expects a single channel (1-D psd)")

    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        raise ValueError(f"No PSD bins fall in {fmin}-{fmax} Hz")

    band_freqs, band_psd = freqs[mask], psd[mask]
    local = int(np.argmax(band_psd))
    peak_freq = float(band_freqs[local])
    peak_psd = float(band_psd[local])

    # Interpolation needs a neighbour either side, so skip it at the band edges.
    if interpolate and 0 < local < len(band_psd) - 1:
        y_minus, y_zero, y_plus = np.log(band_psd[local - 1 : local + 2] + 1e-30)
        denom = y_minus - 2.0 * y_zero + y_plus
        if denom != 0.0:
            # Vertex of the parabola through the three points, in bin units.
            offset = 0.5 * (y_minus - y_plus) / denom
            # A shift beyond half a bin means the fit is not describing a peak.
            if abs(offset) <= 0.5:
                spacing = float(band_freqs[local + 1] - band_freqs[local])
                peak_freq = float(band_freqs[local] + offset * spacing)
                peak_psd = float(np.exp(y_zero - 0.25 * (y_minus - y_plus) * offset))

    return peak_freq, peak_psd


# --------------------------------------------------------------------------
# Signal-to-noise at a known frequency
# --------------------------------------------------------------------------

def snr_at_frequency(
    freqs: np.ndarray,
    psd: np.ndarray,
    f0: float,
    *,
    n_side_bins: int = 10,
    n_guard_bins: int = 1,
) -> float:
    """Signal-to-noise ratio in dB at a known frequency.

    Compares the power in the bin at ``f0`` against the average of nearby bins,
    which stand in for "what the spectrum would look like here if there were no
    response". This is the standard way to quantify a steady-state evoked
    response, where you know the stimulus frequency in advance.

    The guard bins
    --------------
    Bins immediately adjacent to a strong peak are contaminated by it: the Hann
    window's leakage puts real signal power into its neighbours. Including those
    in the noise estimate inflates the noise and understates the SNR, so
    ``n_guard_bins`` on each side are excluded from the noise average while
    still not being counted as signal.

    Parameters
    ----------
    freqs, psd : ndarray
        From :func:`welch_psd`. ``psd`` must be 1-D.
    f0 : float
        Frequency of interest in Hz. Known, not discovered -- searching for the
        best frequency and then reporting its SNR is circular.
    n_side_bins : int
        Noise bins to average on each side. Default 10.
    n_guard_bins : int
        Bins adjacent to the target excluded from the noise estimate. Default 1.

    Returns
    -------
    float
        SNR in dB. 0 dB means the bin at ``f0`` is no stronger than its
        surroundings.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if psd.ndim != 1:
        raise ValueError("snr_at_frequency expects a single channel (1-D psd)")

    target = int(np.argmin(np.abs(freqs - f0)))

    lo_start = max(0, target - n_guard_bins - n_side_bins)
    lo_stop = max(0, target - n_guard_bins)
    hi_start = min(len(psd), target + n_guard_bins + 1)
    hi_stop = min(len(psd), target + n_guard_bins + 1 + n_side_bins)

    noise_bins = np.concatenate([psd[lo_start:lo_stop], psd[hi_start:hi_stop]])
    if noise_bins.size == 0:
        raise ValueError(
            f"No noise bins available around {f0} Hz. The frequency may be too "
            "close to DC or to Nyquist for this window length."
        )

    noise = float(np.mean(noise_bins))
    return float(10.0 * np.log10(psd[target] / (noise + 1e-30) + 1e-30))


def tone_amplitude(x: np.ndarray, fs: float, f0: float) -> float:
    """Peak amplitude of a sinusoid at a known frequency, by coherent detection.

    For a *known* frequency this beats reading a Welch peak. Rather than
    estimating the whole spectrum and looking up a bin, it correlates the signal
    directly against a complex sinusoid at exactly ``f0``:

        X = sum( x[n] * exp(-2*pi*j*f0*n/fs) )

    and the peak amplitude is ``2|X|/N``.

    Why not just use the PSD
    ------------------------
    A Welch bin is centred at ``k*fs/nperseg``, and a tone between two bin
    centres has its power split between them -- "scalloping loss", up to about
    3.9 dB with a Hann window. For a measurement quoted in decibels, such as a
    rejection ratio, that error is unacceptable. Coherent detection has no bin
    grid at all, so no scalloping.

    The signal is truncated to a whole number of cycles of ``f0`` first. A
    partial cycle leaves a discontinuity between the assumed period and the data
    window, which leaks into the estimate.

    Parameters
    ----------
    x : ndarray
        Signal, 1-D, in microvolts (or any unit -- the result matches it).
    fs : float
        Sampling rate in Hz.
    f0 : float
        Frequency of the tone in Hz.

    Returns
    -------
    float
        Peak amplitude of the component at ``f0``, in the units of ``x``.
    """
    x = np.asarray(x, dtype=float).ravel()
    if f0 <= 0:
        raise ValueError("f0 must be positive")

    samples_per_cycle = fs / f0
    n_cycles = int(len(x) // samples_per_cycle)
    if n_cycles < 1:
        raise ValueError(
            f"Signal is shorter than one cycle of {f0} Hz "
            f"({len(x)} samples at {fs} Hz). Capture for longer."
        )
    n = int(round(n_cycles * samples_per_cycle))
    seg = x[:n] - np.mean(x[:n])

    t = np.arange(n) / fs
    dft = np.sum(seg * np.exp(-2j * np.pi * f0 * t))
    return float(2.0 * np.abs(dft) / n)


# --------------------------------------------------------------------------
# Phase locking
# --------------------------------------------------------------------------

def phase_locking_value(
    epochs: np.ndarray,
    fs: float,
    f0: float,
    *,
    bandwidth: float = 1.0,
    at_sample: int = 0,
) -> float:
    """Phase-locking value of a set of epochs to their own time origin.

    Answers a question power cannot: *is the response at a consistent phase
    relative to the stimulus?* Each epoch contributes a unit vector pointing in
    the direction of its phase at ``at_sample``. If the response is driven by
    the stimulus, those vectors align and their mean is long (PLV near 1). If
    the oscillation is present but unrelated to the stimulus, the phases are
    scattered and the mean vector is short (PLV near 0).

        PLV = | mean over epochs of exp(j * phase) |

    Why this matters
    ----------------
    A subject with a strong spontaneous rhythm at 10 Hz will show high power and
    high SNR at 10 Hz whether or not a 10 Hz stimulus is driving anything. Power
    cannot separate the two; phase consistency can. Measuring both, and finding
    high SNR with chance-level PLV, is how you catch that case.

    Note the reference here is the epoch's own time zero -- i.e. the stimulus
    marker -- not another electrode. That matters: comparing the phase of two
    electrodes recorded against a *shared* reference electrode is confounded,
    because whatever the reference picks up appears in both signals. Locking to
    an external marker has no such problem.

    Parameters
    ----------
    epochs : ndarray
        Shape (n_epochs, n_samples). Each row starts at the event of interest.
    fs : float
        Sampling rate in Hz.
    f0 : float
        Frequency to examine, in Hz.
    bandwidth : float
        Half-width of the band-pass around ``f0``, in Hz. Default 1.0, i.e.
        ``f0 +/- 1 Hz``. Narrow enough to isolate the response, wide enough that
        the filter settles within an epoch.
    at_sample : int
        Sample index at which to read each epoch's phase. Default 0, the event.

    Returns
    -------
    float
        PLV in [0, 1]. Compare against :func:`rayleigh_threshold` to judge
        whether a value is above chance for your epoch count.
    """
    epochs = np.atleast_2d(np.asarray(epochs, dtype=float))
    n_epochs, n_samples = epochs.shape
    if n_epochs < 2:
        raise ValueError("PLV needs at least 2 epochs")

    low, high = f0 - bandwidth, f0 + bandwidth
    if low <= 0:
        raise ValueError(f"f0 ({f0}) must exceed bandwidth ({bandwidth})")
    if high >= fs / 2:
        raise ValueError(f"f0 + bandwidth ({high}) reaches Nyquist ({fs / 2})")

    # Zero-phase band-pass. filtfilt runs the filter forwards then backwards, so
    # the phase distortion of the forward pass is undone by the reverse pass.
    # A one-directional filter would impose its own delay on the phase, which is
    # precisely the quantity being measured here.
    sos = sp_signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = sp_signal.sosfiltfilt(sos, epochs, axis=-1)

    # The analytic signal's angle is the instantaneous phase.
    phase = np.angle(sp_signal.hilbert(filtered, axis=-1))

    if not -n_samples <= at_sample < n_samples:
        raise ValueError(f"at_sample {at_sample} is outside the epoch")

    return float(np.abs(np.mean(np.exp(1j * phase[:, at_sample]))))


def rayleigh_threshold(n_epochs: int, alpha: float = 0.05) -> float:
    """PLV a set of random phases would exceed with probability ``alpha``.

    Phase locking is never exactly zero with finite data: n random directions
    have a mean vector of length roughly ``1/sqrt(n)`` just by chance. This
    returns the value to beat, from the Rayleigh test for circular uniformity:

        threshold = sqrt( -ln(alpha) / n )

    So 10 epochs give a 5% threshold of about 0.55 -- a PLV of 0.5 from 10
    epochs is *not* evidence of anything. Always report this alongside a PLV, or
    the number cannot be interpreted.

    Parameters
    ----------
    n_epochs : int
        Number of epochs the PLV was computed over.
    alpha : float
        Significance level. Default 0.05.

    Returns
    -------
    float
        The chance-level PLV threshold.
    """
    if n_epochs < 1:
        raise ValueError("n_epochs must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    threshold = float(np.sqrt(-np.log(alpha) / n_epochs))
    if threshold >= 1.0:
        warnings.warn(
            f"With {n_epochs} epochs the {alpha:.0%} chance threshold is "
            f"{threshold:.2f}, at or above the maximum PLV of 1.0. No PLV from "
            "this many epochs can be called significant. Record more trials.",
            stacklevel=2,
        )
    return threshold
