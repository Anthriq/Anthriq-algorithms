"""
End-to-end tests: generate data with a known answer, analyse it, check the
answer comes back.

This is the test that matters most to someone trusting these scripts. The unit
tests show each function is correct in isolation; these show the whole path from
a file on disk to a reported number does not lose or distort anything along the
way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import alpha as alpha_script  # noqa: E402
import cmrr as cmrr_script  # noqa: E402
import ssvep as ssvep_script  # noqa: E402
import synth  # noqa: E402

from exg.io import load  # noqa: E402


# --------------------------------------------------------------------------
# Alpha
# --------------------------------------------------------------------------

def test_alpha_recovers_the_generated_effect(tmp_path):
    """A 4x amplitude difference is a 16x power difference."""
    folder, truth = synth.make_alpha(
        tmp_path / "a", amp_closed=20.0, amp_open=5.0, noise_uv=1.0, seed=0
    )
    results = alpha_script.analyse(load(folder), ["O1"])

    assert truth["expected_ratio_signal_only"] == 16.0
    assert results["reactivity_ratio"] == pytest.approx(16.0, rel=0.15)
    assert results["reactivity_normalised"] > 0.8
    assert results["peak_frequency_hz"] == pytest.approx(truth["alpha_freq_hz"], abs=0.1)


def test_alpha_power_matches_the_generated_amplitude(tmp_path):
    """One channel at unit weight must return A**2/2 for its own amplitude."""
    folder, _ = synth.make_alpha(
        tmp_path / "a", amp_closed=20.0, amp_open=5.0, noise_uv=0.5, seed=1
    )
    # O1 carries weight 1.0 in the generator, so its power is exactly A**2/2.
    results = alpha_script.analyse(load(folder), ["O1"])
    assert results["alpha_power_closed_uv2"] == pytest.approx(200.0, rel=0.1)
    assert results["alpha_power_open_uv2"] == pytest.approx(12.5, rel=0.25)


def test_alpha_detects_an_inverted_effect(tmp_path):
    """Swapped conditions must produce a negative index, not a plausible one.

    Mislabelled conditions are far more common than a subject whose alpha rises
    when they open their eyes, so the sign has to be trustworthy: it is the
    thing that tells you to go and check the labels.
    """
    folder, _ = synth.make_alpha(
        tmp_path / "a", amp_closed=5.0, amp_open=20.0, noise_uv=1.0, seed=2
    )
    results = alpha_script.analyse(load(folder), ["O1"])
    assert results["reactivity_normalised"] < -0.5
    assert results["reactivity_ratio"] < 1.0


def test_alpha_reports_no_peak_when_there_is_no_rhythm(tmp_path):
    """Pink noise alone has no peak; the prominence test must say so.

    Returning the argmax of a 1/f slope would be a number with no meaning, and
    it would look just like a real individual peak frequency.
    """
    folder, _ = synth.make_alpha(
        tmp_path / "a", amp_closed=0.0, amp_open=0.0, noise_uv=10.0, seed=3
    )
    results = alpha_script.analyse(load(folder), ["O1"])
    assert not np.isfinite(results["peak_frequency_hz"])


def test_alpha_is_weak_at_the_frontal_control_site(tmp_path):
    """The generator puts a tenth of the rhythm at Fpz, as real heads do."""
    folder, _ = synth.make_alpha(tmp_path / "a", noise_uv=1.0, seed=4)
    recording = load(folder)
    occipital = alpha_script.analyse(recording, ["O1"])
    frontal = alpha_script.analyse(recording, ["Fpz"])
    assert frontal["alpha_power_closed_uv2"] < occipital["alpha_power_closed_uv2"] / 10


def test_alpha_accepts_explicit_windows_instead_of_markers(tmp_path):
    folder, _ = synth.make_alpha(
        tmp_path / "a", closed_seconds=30.0, open_seconds=15.0, noise_uv=1.0, seed=5
    )
    results = alpha_script.analyse(
        load(folder), ["O1"],
        closed_marker="absent", open_marker="absent",
        closed_window=(1.0, 29.0), open_window=(31.0, 44.0),
    )
    assert results["reactivity_ratio"] > 4.0


def test_alpha_explains_itself_when_a_condition_cannot_be_found(tmp_path):
    folder, _ = synth.make_alpha(tmp_path / "a", seed=6)
    with pytest.raises(ValueError, match="markers are"):
        alpha_script.analyse(load(folder), ["O1"], closed_marker="nope", open_marker="nope")


# --------------------------------------------------------------------------
# SSVEP
# --------------------------------------------------------------------------

def test_ssvep_finds_the_response_at_the_stimulus_frequency(tmp_path):
    folder, truth = synth.make_ssvep(
        tmp_path / "s", stim_freq=12.0, noise_uv=2.0, seed=0
    )
    results = ssvep_script.analyse(load(folder), ["O1"], 12.0)

    assert results["peak_frequency_hz"] == pytest.approx(12.0, abs=0.1)
    assert results["harmonics"][0]["snr_flicker_db"] > 15.0
    # The generator emits a second harmonic, so it must be found too.
    assert results["harmonics"][1]["snr_flicker_db"] > 10.0
    # Nothing was generated at 3f.
    assert results["harmonics"][2]["snr_flicker_db"] < 6.0
    assert results["power_ratio"] > 10.0


def test_ssvep_phase_locking_is_high_when_phases_align(tmp_path):
    folder, _ = synth.make_ssvep(
        tmp_path / "s", stim_freq=12.0, n_trials=20, phase_jitter=0.0, seed=1
    )
    results = ssvep_script.analyse(load(folder), ["O1"], 12.0)
    assert results["plv"] > 0.9
    assert results["plv"] > results["plv_chance_95"]


def test_ssvep_high_power_with_chance_phase_locking(tmp_path):
    """The decoy: a rhythm at the stimulus frequency that is not driven by it.

    This is the case that justifies measuring phase at all. Power says there is
    a strong response; phase says nothing is locked to the stimulus. A pipeline
    reporting only power would call this an SSVEP.
    """
    folder, _ = synth.make_ssvep(
        tmp_path / "s", stim_freq=12.0, n_trials=30,
        phase_jitter=2 * np.pi, noise_uv=2.0, seed=2,
    )
    results = ssvep_script.analyse(load(folder), ["O1"], 12.0)

    assert results["harmonics"][0]["snr_flicker_db"] > 15.0   # power: strong
    assert results["plv"] < results["plv_chance_95"]          # phase: nothing


def test_ssvep_response_is_weaker_at_the_control_site(tmp_path):
    folder, _ = synth.make_ssvep(tmp_path / "s", stim_freq=12.0, noise_uv=2.0, seed=3)
    results = ssvep_script.analyse(load(folder), ["O1"], 12.0, control=["Fpz"])
    assert results["control_snr_db"] < results["harmonics"][0]["snr_flicker_db"] - 6.0


def test_ssvep_tracks_whichever_frequency_was_used(tmp_path):
    """The peak must follow the stimulus, which is what rules out interference."""
    for frequency in (10.0, 15.0):
        folder, _ = synth.make_ssvep(
            tmp_path / f"s{frequency:g}", stim_freq=frequency, noise_uv=2.0, seed=4
        )
        results = ssvep_script.analyse(load(folder), ["O1"], frequency)
        assert results["peak_frequency_hz"] == pytest.approx(frequency, abs=0.15)


def test_ssvep_measured_at_the_wrong_frequency_finds_nothing(tmp_path):
    """Asking about 20 Hz when 12 Hz was presented must report no response.

    A pipeline that searched for its own best frequency would pass this test by
    finding the 12 Hz peak anyway, which is why the frequency is an argument.
    """
    folder, _ = synth.make_ssvep(tmp_path / "s", stim_freq=12.0, noise_uv=2.0, seed=5)
    results = ssvep_script.analyse(load(folder), ["O1"], 20.0)
    assert results["harmonics"][0]["snr_flicker_db"] < 6.0


def test_ssvep_needs_markers_for_phase_locking(tmp_path):
    folder, _ = synth.make_ssvep(tmp_path / "s", stim_freq=12.0, seed=6)
    recording = load(folder)
    recording.markers = [m for m in recording.markers if m.name != "stimulus"]
    with pytest.raises(ValueError, match="No spans found"):
        ssvep_script.analyse(recording, ["O1"], 12.0)


# --------------------------------------------------------------------------
# CMRR
# --------------------------------------------------------------------------

def test_cmrr_recovers_the_generated_sweep(tmp_path):
    folder, truth = synth.make_cmrr(tmp_path / "c", gain=100.0, seed=0)
    results = cmrr_script.analyse(folder, fs=truth["fs_hz"], monitor=truth["monitor_channel"])

    for measurement, worst in zip(results["measurements"], results["worst_case_db"]):
        expected = truth["expected_cmrr_db"][measurement["frequency_hz"]]
        assert worst == pytest.approx(expected, abs=0.5)


def test_cmrr_is_the_same_at_any_gain(tmp_path):
    """The point of the node-referenced method, as an executable check.

    The monitor and the channels share a gain stage, so the gain cancels in the
    ratio. Measuring the same rejection at two very different gains is the
    cheapest possible confirmation that the rig is wired as intended -- if the
    curves differ, the monitor is not in the same node.
    """
    curves = []
    for gain in (10.0, 2000.0):
        folder, truth = synth.make_cmrr(tmp_path / f"c{gain:g}", gain=gain, seed=1)
        results = cmrr_script.analyse(
            folder, fs=truth["fs_hz"], monitor=truth["monitor_channel"]
        )
        curves.append(results["worst_case_db"])

    assert curves[0] == pytest.approx(curves[1], abs=0.1)


def test_cmrr_rises_with_frequency_as_generated(tmp_path):
    folder, truth = synth.make_cmrr(tmp_path / "c", seed=2)
    results = cmrr_script.analyse(folder, fs=truth["fs_hz"], monitor=truth["monitor_channel"])
    worst = results["worst_case_db"]
    assert worst == sorted(worst)


def test_cmrr_skips_a_capture_with_no_drive_tone(tmp_path):
    """A capture recorded with the generator off is not a measurement of zero.

    The ratio of two noise floors is a meaningless number that would otherwise
    join the sweep looking like data.
    """
    folder, truth = synth.make_cmrr(tmp_path / "c", seed=3)
    rng = np.random.default_rng(0)
    dead = rng.standard_normal((8000, 4)) * 1e-6
    np.savetxt(
        folder / "capture_75Hz.csv", dead, delimiter=",",
        header="Mod_9234/ai0,Mod_9234/ai1,Mod_9234/ai2,Mod_9234/ai3", comments="",
    )

    results = cmrr_script.analyse(folder, fs=truth["fs_hz"], monitor=truth["monitor_channel"])
    assert 75.0 not in results["frequencies_hz"]
    assert any("75Hz" in name for name, _ in results["skipped"])


@pytest.mark.parametrize("filename,expected", [
    ("capture_50Hz.csv", 50.0),
    ("cap-12.5hz.csv", 12.5),
    ("sweep 100 Hz.csv", 100.0),
    ("run_1.5Hz.csv", 1.5),
])
def test_cmrr_reads_the_frequency_from_the_filename(tmp_path, filename, expected):
    assert cmrr_script.frequency_from_name(Path(filename)) == expected


def test_cmrr_ignores_a_file_with_no_frequency_in_its_name(tmp_path):
    folder, truth = synth.make_cmrr(tmp_path / "c", seed=4)
    (folder / "notes.csv").write_text("Mod_9234/ai0\n0.1\n")
    results = cmrr_script.analyse(folder, fs=truth["fs_hz"], monitor=truth["monitor_channel"])
    assert any("notes.csv" in name for name, _ in results["skipped"])


def test_cmrr_refuses_to_extrapolate_past_the_sweep(tmp_path):
    """Interpolating inside the measured range is fine; beyond it is a guess."""
    frequencies = [1.0, 10.0, 100.0]
    values = [10.0, 30.0, 50.0]
    assert cmrr_script.interpolate_at(frequencies, values, 50.0) is not None
    assert cmrr_script.interpolate_at(frequencies, values, 500.0) is None
    assert cmrr_script.interpolate_at(frequencies, values, 0.5) is None


def test_cmrr_needs_a_directory(tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("ai0\n0.1\n")
    with pytest.raises(NotADirectoryError, match="one capture"):
        cmrr_script.analyse(path, fs=2000.0, monitor="ai0")


# --------------------------------------------------------------------------
# The generators themselves
# --------------------------------------------------------------------------

def test_generators_are_deterministic(tmp_path):
    """Same seed, same bytes. Without this, a failing test cannot be reproduced."""
    first, _ = synth.make_alpha(tmp_path / "one", seed=42)
    second, _ = synth.make_alpha(tmp_path / "two", seed=42)
    assert (first / "one.csv").read_bytes() == (second / "two.csv").read_bytes()


def test_generated_exports_round_trip_through_the_reader(tmp_path):
    folder, _ = synth.make_alpha(tmp_path / "a", fs=250.0, seed=7)
    recording = load(folder)
    assert recording.fs == 250.0
    assert recording.channels == ["O1", "O2", "Fpz"]
    assert set(recording.marker_names()) == {"eyes_closed", "eyes_open"}


def test_pink_noise_falls_with_frequency(tmp_path):
    """The background must be 1/f, not white: white noise flatters every analysis."""
    from exg.spectra import band_power, welch_psd

    rng = np.random.default_rng(0)
    noise = synth.pink_noise(100_000, 500.0, rng, exponent=1.0)
    freqs, psd = welch_psd(noise, 500.0)
    low = band_power(freqs, psd, 2.0, 6.0)
    high = band_power(freqs, psd, 40.0, 44.0)
    assert low > high * 4
