"""
Tests for the spectral functions, against answers known in closed form.

The point of these tests is not coverage. It is that every number the analyses
report can be traced back to a case where the right answer is known
independently of the code -- a sinusoid of known amplitude, a set of phases with
known alignment. A pipeline that passes these is measuring what it claims to.
"""

from __future__ import annotations

import numpy as np
import pytest

from exg.spectra import (
    band_power,
    choose_nperseg,
    peak_in_band,
    phase_locking_value,
    rayleigh_threshold,
    snr_at_frequency,
    tone_amplitude,
    welch_psd,
)

FS = 250.0
DURATION = 60.0


def sinusoid(amplitude: float, frequency: float, fs: float = FS,
             duration: float = DURATION, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(fs * duration)) / fs
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


# --------------------------------------------------------------------------
# Band power: the anchor for every amplitude in every analysis
# --------------------------------------------------------------------------

@pytest.mark.parametrize("amplitude", [0.5, 1.0, 20.0, 100.0, 500.0])
def test_band_power_of_a_sinusoid_is_half_amplitude_squared(amplitude):
    """A sine of peak amplitude A has mean-square power A**2/2.

    This is the single most important assertion in the suite: it fixes the unit
    convention end to end. If it passes at several amplitudes, the whole chain
    from samples to uV**2 is linear and correctly scaled.
    """
    freqs, psd = welch_psd(sinusoid(amplitude, 10.0), FS)
    assert band_power(freqs, psd, 8.0, 12.0) == pytest.approx(amplitude ** 2 / 2, rel=0.01)


def test_band_power_is_negligible_away_from_the_signal():
    freqs, psd = welch_psd(sinusoid(20.0, 10.0), FS)
    in_band = band_power(freqs, psd, 8.0, 12.0)
    out_of_band = band_power(freqs, psd, 20.0, 30.0)
    assert out_of_band < in_band / 1000.0


def test_band_power_over_the_whole_spectrum_matches_the_variance():
    """Parseval's theorem, as an independent check on the integration.

    Total power across all frequencies must equal the signal's variance. This
    validates the density-to-power integration without reference to the
    sinusoid case, so the two tests fail independently.
    """
    rng = np.random.default_rng(0)
    x = sinusoid(20.0, 10.0) + 5.0 * rng.standard_normal(int(FS * DURATION))
    freqs, psd = welch_psd(x, FS)
    assert band_power(freqs, psd, 0.0, FS / 2) == pytest.approx(np.var(x), rel=0.05)


def test_a_single_bin_band_is_not_silently_zero():
    """Trapezoidal integration over one point is zero; that must not happen.

    A band narrower than the bin spacing has to return its density times the
    bin width. Returning 0.0 would be indistinguishable from a real measurement
    of no power, which is the kind of bug that survives review.
    """
    freqs, psd = welch_psd(sinusoid(20.0, 10.0), FS, nperseg=int(4 * FS))
    spacing = freqs[1] - freqs[0]
    narrow = band_power(freqs, psd, 10.0, 10.0 + spacing * 0.5)
    assert narrow > 0.0


def test_a_band_between_bins_raises_rather_than_returning_zero():
    freqs, psd = welch_psd(sinusoid(20.0, 10.0), FS, nperseg=64)
    with pytest.raises(ValueError, match="No PSD bins"):
        band_power(freqs, psd, 10.001, 10.002)


def test_band_power_rejects_an_inverted_band():
    freqs, psd = welch_psd(sinusoid(20.0, 10.0), FS)
    with pytest.raises(ValueError, match="must exceed"):
        band_power(freqs, psd, 12.0, 8.0)


# --------------------------------------------------------------------------
# Peak location
# --------------------------------------------------------------------------

@pytest.mark.parametrize("frequency", [8.7, 10.0, 10.2, 11.43])
def test_peak_is_found_between_bins(frequency):
    """Interpolation must beat the bin grid.

    With 0.25 Hz bins, a peak at 11.43 Hz would be reported as 11.5 without
    interpolation. Individual peak alpha frequency is exactly the kind of
    measurement where that error matters, since the whole point is comparing
    one person's value with another's.
    """
    freqs, psd = welch_psd(sinusoid(20.0, frequency), FS, nperseg=int(4 * FS))
    found, _ = peak_in_band(freqs, psd, 7.0, 13.0)
    assert found == pytest.approx(frequency, abs=0.05)


def test_peak_without_interpolation_snaps_to_a_bin():
    freqs, psd = welch_psd(sinusoid(20.0, 11.43), FS, nperseg=int(4 * FS))
    found, _ = peak_in_band(freqs, psd, 7.0, 13.0, interpolate=False)
    spacing = freqs[1] - freqs[0]
    assert found % spacing == pytest.approx(0.0, abs=1e-9)


def test_peak_needs_a_single_channel():
    freqs, psd = welch_psd(np.vstack([sinusoid(20.0, 10.0)] * 2), FS)
    with pytest.raises(ValueError, match="single channel"):
        peak_in_band(freqs, psd, 8.0, 12.0)


# --------------------------------------------------------------------------
# Signal-to-noise
# --------------------------------------------------------------------------

def test_snr_is_high_on_signal_and_near_zero_on_noise():
    rng = np.random.default_rng(1)
    x = sinusoid(10.0, 12.0) + 5.0 * rng.standard_normal(int(FS * DURATION))
    freqs, psd = welch_psd(x, FS)
    assert snr_at_frequency(freqs, psd, 12.0) > 20.0
    # Nothing was put at 30 Hz, so the bin there should look like its neighbours.
    assert abs(snr_at_frequency(freqs, psd, 30.0)) < 4.0


def test_snr_rises_with_amplitude():
    rng = np.random.default_rng(2)
    noise = 5.0 * rng.standard_normal(int(FS * DURATION))
    values = []
    for amplitude in (2.0, 10.0, 50.0):
        freqs, psd = welch_psd(sinusoid(amplitude, 12.0) + noise, FS)
        values.append(snr_at_frequency(freqs, psd, 12.0))
    assert values[0] < values[1] < values[2]


# --------------------------------------------------------------------------
# Coherent tone detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("frequency", [10.0, 10.137, 49.6, 50.0])
def test_tone_amplitude_is_exact_regardless_of_the_bin_grid(frequency):
    """Coherent detection must not suffer scalloping loss.

    A tone sitting between two FFT bins loses up to 3.9 dB from a windowed
    periodogram. For a measurement quoted in dB -- a rejection ratio, say --
    that error is unacceptable, which is why CMRR uses this rather than a peak.
    """
    measured = tone_amplitude(sinusoid(50.0, frequency), FS, frequency)
    assert measured == pytest.approx(50.0, rel=0.01)


def test_tone_amplitude_ignores_a_dc_offset():
    x = sinusoid(50.0, 10.0) + 1000.0
    assert tone_amplitude(x, FS, 10.0) == pytest.approx(50.0, rel=0.01)


def test_tone_amplitude_refuses_a_signal_shorter_than_one_cycle():
    with pytest.raises(ValueError, match="shorter than one cycle"):
        tone_amplitude(np.zeros(10), FS, 1.0)


# --------------------------------------------------------------------------
# Phase locking
# --------------------------------------------------------------------------

def make_epochs(n_epochs: int, frequency: float, jitter: float,
                seed: int = 0, noise: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(2 * FS)
    return np.array([
        sinusoid(10.0, frequency, duration=2.0,
                 phase=rng.uniform(-jitter / 2, jitter / 2))
        + noise * rng.standard_normal(n)
        for _ in range(n_epochs)
    ])


def test_identical_phases_give_plv_near_one():
    plv = phase_locking_value(make_epochs(40, 12.0, jitter=0.0), FS, 12.0)
    assert plv > 0.95


def test_scattered_phases_give_plv_near_chance():
    n_epochs = 60
    plv = phase_locking_value(make_epochs(n_epochs, 12.0, jitter=2 * np.pi), FS, 12.0)
    assert plv < 3.0 / np.sqrt(n_epochs)


def test_plv_falls_as_jitter_grows():
    values = [
        phase_locking_value(make_epochs(40, 12.0, jitter=j, seed=3), FS, 12.0)
        for j in (0.0, np.pi / 2, np.pi, 2 * np.pi)
    ]
    assert values == sorted(values, reverse=True)


def test_high_snr_with_chance_plv_is_detectable():
    """The case that justifies measuring phase at all.

    A rhythm at the stimulus frequency with random phase per trial produces
    strong power and no phase locking. Power alone would call this a response;
    the pair of measures does not.
    """
    epochs = make_epochs(60, 12.0, jitter=2 * np.pi, seed=4)
    freqs, psd = welch_psd(epochs.reshape(-1), FS)
    assert snr_at_frequency(freqs, psd, 12.0) > 10.0
    plv = phase_locking_value(epochs, FS, 12.0)
    assert plv < rayleigh_threshold(60)


def test_plv_needs_at_least_two_epochs():
    with pytest.raises(ValueError, match="at least 2"):
        phase_locking_value(make_epochs(1, 12.0, jitter=0.0), FS, 12.0)


def test_plv_rejects_a_band_reaching_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        phase_locking_value(make_epochs(4, 12.0, jitter=0.0), FS, FS / 2 - 0.5)


# --------------------------------------------------------------------------
# The chance threshold
# --------------------------------------------------------------------------

def test_rayleigh_threshold_falls_with_more_epochs():
    assert rayleigh_threshold(10) > rayleigh_threshold(100) > rayleigh_threshold(1000)


def test_rayleigh_threshold_matches_the_closed_form():
    assert rayleigh_threshold(100, 0.05) == pytest.approx(np.sqrt(-np.log(0.05) / 100))


def test_rayleigh_threshold_warns_when_no_plv_could_be_significant():
    with pytest.warns(UserWarning, match="No PLV"):
        rayleigh_threshold(2)


# --------------------------------------------------------------------------
# Window selection
# --------------------------------------------------------------------------

def test_nperseg_scales_with_the_sampling_rate():
    """A window is a duration, not a sample count.

    The same recording length in seconds must give the same window in seconds
    at any sampling rate. Hard-coding a sample count is the mistake this
    function exists to prevent.
    """
    at_250 = choose_nperseg(int(250 * 30), 250.0) / 250.0
    at_4000 = choose_nperseg(int(4000 * 30), 4000.0) / 4000.0
    assert at_250 == pytest.approx(at_4000, rel=0.01)


def test_nperseg_is_clamped_to_the_stated_range():
    assert choose_nperseg(int(250 * 3600), 250.0) <= int(8.0 * 250)
    assert choose_nperseg(int(250 * 2), 250.0) >= int(1.0 * 250)


def test_nperseg_never_exceeds_the_signal():
    assert choose_nperseg(100, 250.0) <= 100


def test_welch_accepts_multichannel_input():
    data = np.vstack([sinusoid(20.0, 10.0), sinusoid(10.0, 10.0)])
    freqs, psd = welch_psd(data, FS)
    assert psd.shape[0] == 2
    powers = band_power(freqs, psd, 8.0, 12.0)
    # Halving the amplitude quarters the power.
    assert powers[0] / powers[1] == pytest.approx(4.0, rel=0.02)
