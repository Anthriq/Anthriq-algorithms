# Copyright 2026 Nexstem India Private Limited (trading as Anthriq)
# Licensed under the Apache License, Version 2.0. See the LICENSE file.

"""
Tests for reading BIDS datasets and BrainVision files.

The important assertion is the last group: an analysis must give the same
answer whether it read a BIDS dataset or a plain CSV. A reader that silently
scales, reorders or mistimes anything would show up there rather than in a
parsing test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import alpha as alpha_script  # noqa: E402
import synth  # noqa: E402

from exg.bids import (  # noqa: E402
    describe_bids_dataset,
    find_bids_recordings,
    read_bids_recording,
    read_brainvision,
)
from exg.io import load  # noqa: E402

FS = 250.0


def make_dataset(tmp_path, *, n_channels=2, n_samples=2500, fs=FS, task="test",
                 amplitude=50.0, markers=None):
    """Write a small BIDS dataset with a known signal in it."""
    t = np.arange(n_samples) / fs
    data = np.vstack([
        amplitude * (index + 1) * np.sin(2 * np.pi * 10.0 * t)
        for index in range(n_channels)
    ])
    return synth.write_bids_dataset(
        tmp_path / "ds", data, fs,
        [f"E{i + 1}" for i in range(n_channels)],
        markers if markers is not None else [("rest", 1.0, 2.0)],
        task=task,
    ), data


# --------------------------------------------------------------------------
# BrainVision
# --------------------------------------------------------------------------

def test_header_magic_line_does_not_break_parsing(tmp_path):
    """A .vhdr opens with a magic line before any [Section].

    Every real BrainVision header has it, and a plain INI parser rejects the
    file outright, so this is the first thing a reader has to handle.
    """
    root, _ = make_dataset(tmp_path)
    vhdr = next(root.glob("sub-*/eeg/*_eeg.vhdr"))
    assert vhdr.read_text().startswith("Brain Vision Data Exchange")
    data, names, fs, _ = read_brainvision(vhdr)
    assert fs == pytest.approx(FS)
    assert names == ["E1", "E2"]


def test_sampling_rate_comes_from_the_interval(tmp_path):
    """The header states microseconds between samples, not a rate."""
    root, _ = make_dataset(tmp_path, fs=1000.0)
    vhdr = next(root.glob("sub-*/eeg/*_eeg.vhdr"))
    assert "SamplingInterval=1000" in vhdr.read_text()
    _, _, fs, _ = read_brainvision(vhdr)
    assert fs == pytest.approx(1000.0)


def test_samples_are_recovered_exactly(tmp_path):
    """Multiplexed layout means channels interleave sample by sample.

    Reading them as if they were stored channel-by-channel would give a signal
    that looks plausible but is the wrong length per channel and wrong in time.
    """
    root, original = make_dataset(tmp_path, n_channels=3, amplitude=25.0)
    vhdr = next(root.glob("sub-*/eeg/*_eeg.vhdr"))
    data, _, _, _ = read_brainvision(vhdr)
    assert data.shape == original.shape
    # float32 storage, so exact to about seven significant figures.
    assert np.allclose(data, original, rtol=1e-5, atol=1e-4)


def test_channels_are_not_transposed(tmp_path):
    """Each channel is generated at a different amplitude, so a swap shows."""
    root, original = make_dataset(tmp_path, n_channels=3, amplitude=10.0)
    data, _, _, _ = read_brainvision(next(root.glob("sub-*/eeg/*_eeg.vhdr")))
    for index in range(3):
        assert np.max(np.abs(data[index])) == pytest.approx(
            np.max(np.abs(original[index])), rel=0.01
        )


def test_a_file_without_sections_is_rejected(tmp_path):
    bad = tmp_path / "bad.vhdr"
    bad.write_text("Brain Vision Data Exchange Header File Version 1.0\n")
    with pytest.raises(ValueError, match="no \\[Section\\]"):
        read_brainvision(bad)


def test_a_missing_data_file_says_which_one(tmp_path):
    root, _ = make_dataset(tmp_path)
    vhdr = next(root.glob("sub-*/eeg/*_eeg.vhdr"))
    next(root.glob("sub-*/eeg/*_eeg.eeg")).unlink()
    with pytest.raises(FileNotFoundError, match="three files that travel together"):
        read_brainvision(vhdr)


# --------------------------------------------------------------------------
# The BIDS layer
# --------------------------------------------------------------------------

def test_events_become_markers_with_seconds_preserved(tmp_path):
    """BIDS states onset and duration in seconds, not samples."""
    markers = [("eyes_closed", 2.5, 30.0), ("eyes_open", 32.5, 10.0)]
    root, _ = make_dataset(tmp_path, n_samples=int(FS * 60), markers=markers)
    recording = read_bids_recording(next(root.glob("sub-*/eeg/*_eeg.vhdr")))

    assert [m.name for m in recording.markers] == ["eyes_closed", "eyes_open"]
    assert [m.onset for m in recording.markers] == pytest.approx([2.5, 32.5])
    assert [m.duration for m in recording.markers] == pytest.approx([30.0, 10.0])


def test_trigger_channels_are_kept_out_of_the_signal(tmp_path):
    """A trigger channel is not a biosignal and must not join an ROI average.

    Averaging a square-wave trigger into an occipital pair would add broadband
    power at the stimulus rate -- which looks exactly like a response.
    """
    t = np.arange(1000) / FS
    data = np.vstack([
        20.0 * np.sin(2 * np.pi * 10 * t),
        20.0 * np.sin(2 * np.pi * 10 * t),
        (np.sin(2 * np.pi * 17 * t) > 0).astype(float) * 1000.0,
    ])
    root = synth.write_bids_dataset(
        tmp_path / "ds", data, FS, ["O1", "O2", "StimTrig"],
        [("flicker", 1.0, 2.0)], task="ssvep",
        channel_types=["EEG", "EEG", "TRIG"],
    )
    recording = read_bids_recording(next(root.glob("sub-*/eeg/*_eeg.vhdr")))

    assert recording.channels == ["O1", "O2"]
    assert "StimTrig" in recording.digital


def test_a_declared_rate_that_disagrees_with_the_header_warns(tmp_path):
    root, _ = make_dataset(tmp_path)
    sidecar = next(root.glob("sub-*/eeg/*_eeg.json"))
    meta = json.loads(sidecar.read_text())
    meta["SamplingFrequency"] = 500.0
    sidecar.write_text(json.dumps(meta))

    with pytest.warns(UserWarning, match="declares 500"):
        recording = read_bids_recording(next(root.glob("sub-*/eeg/*_eeg.vhdr")))
    # The header wins: it describes the file that holds the samples.
    assert recording.fs == pytest.approx(FS)


def test_a_dataset_without_sidecars_still_loads(tmp_path):
    """Only the signal files are strictly needed to read the data."""
    root, _ = make_dataset(tmp_path)
    eeg_dir = next(root.glob("sub-*/eeg"))
    for pattern in ("*_channels.tsv", "*_events.tsv", "*_eeg.json"):
        for path in eeg_dir.glob(pattern):
            path.unlink()
    recording = read_bids_recording(next(eeg_dir.glob("*_eeg.vhdr")))
    assert recording.fs == pytest.approx(FS)
    assert len(recording.channels) == 2


def test_finding_and_describing_a_dataset(tmp_path):
    root, _ = make_dataset(tmp_path)
    assert len(find_bids_recordings(root)) == 1

    description = describe_bids_dataset(root)
    assert description["bids_version"]
    assert description["participants"][0]["participant_id"] == "sub-01"
    assert len(description["recordings"]) == 1


def test_n_a_in_a_duration_becomes_an_instant(tmp_path):
    """BIDS writes n/a for a value it does not have."""
    root, _ = make_dataset(tmp_path)
    events = next(root.glob("sub-*/eeg/*_events.tsv"))
    events.write_text("onset\tduration\ttrial_type\n5.0\tn/a\tblink\n")
    recording = read_bids_recording(next(root.glob("sub-*/eeg/*_eeg.vhdr")))
    assert recording.markers[0].duration == 0.0
    assert recording.markers[0].onset == pytest.approx(5.0)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def test_load_accepts_a_dataset_root(tmp_path):
    root, _ = make_dataset(tmp_path)
    assert load(root).fs == pytest.approx(FS)


def test_load_accepts_a_vhdr_directly(tmp_path):
    root, _ = make_dataset(tmp_path)
    assert load(next(root.glob("sub-*/eeg/*_eeg.vhdr"))).fs == pytest.approx(FS)


def test_load_names_what_it_looked_for_when_a_folder_has_neither(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    (empty / "data.csv").write_text("a,b\n1,2\n")
    with pytest.raises(FileNotFoundError, match="no meta.json.*no \\*_eeg.vhdr"):
        load(empty)


# --------------------------------------------------------------------------
# The assertion that matters: both formats give the same answer
# --------------------------------------------------------------------------

def test_the_same_recording_analyses_identically_from_bids_and_csv(tmp_path):
    """A format change must not move a result.

    The two datasets are generated from the same seed, so the signals are
    identical; only the container differs. Any discrepancy beyond float32
    rounding would mean the reader is scaling, reordering or mistiming
    something.
    """
    from_csv, _ = synth.make_alpha(tmp_path / "csv", layout="bxi", seed=11)
    from_bids, _ = synth.make_alpha(tmp_path / "bids", layout="bids", seed=11)

    csv_result = alpha_script.analyse(load(from_csv), ["O1", "O2"])
    bids_result = alpha_script.analyse(load(from_bids), ["O1", "O2"])

    assert bids_result["alpha_power_closed_uv2"] == pytest.approx(
        csv_result["alpha_power_closed_uv2"], rel=0.01
    )
    assert bids_result["reactivity_ratio"] == pytest.approx(
        csv_result["reactivity_ratio"], rel=0.02
    )
    assert bids_result["peak_frequency_hz"] == pytest.approx(
        csv_result["peak_frequency_hz"], abs=0.02
    )


def test_bids_alpha_recovers_the_generated_ground_truth(tmp_path):
    """The end-to-end check, on the container the sample data will ship in."""
    folder, truth = synth.make_alpha(
        tmp_path / "ds", layout="bids", amp_closed=20.0, amp_open=5.0,
        noise_uv=1.0, seed=12,
    )
    results = alpha_script.analyse(load(folder), ["O1"])
    assert results["reactivity_ratio"] == pytest.approx(16.0, rel=0.15)
    assert results["peak_frequency_hz"] == pytest.approx(truth["alpha_freq_hz"], abs=0.1)


def test_a_byte_order_mark_does_not_hide_the_events(tmp_path):
    """Some BIDS writers prefix a TSV with a UTF-8 byte-order mark.

    Read as plain UTF-8, that mark lands on the front of the first column
    name, so "onset" arrives as "﻿onset". Every lookup of "onset" then
    returns nothing and the events load as an empty list -- no error, no
    warning, just a recording that appears to have no events. This was found
    against a real mne-bids export, not against a fixture.
    """
    root, _ = make_dataset(tmp_path, markers=[("rest", 1.0, 2.0)])
    events = next(root.glob("sub-*/eeg/*_events.tsv"))
    events.write_bytes(b"\xef\xbb\xbf" + events.read_bytes())

    recording = read_bids_recording(next(root.glob("sub-*/eeg/*_eeg.vhdr")))
    assert len(recording.markers) == 1
    assert recording.markers[0].name == "rest"
    assert recording.markers[0].onset == pytest.approx(1.0)
