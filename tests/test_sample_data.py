# Copyright 2026 Nexstem India Private Limited (trading as Anthriq)
# Licensed under the Apache License, Version 2.0. See the LICENSE file.

"""
Tests against the real recordings shipped in examples/sample-data.

Synthetic data proves the maths is right. These prove the whole path works on
a real recording, with real electrode noise, real drift, and a stimulus that
did not do exactly what it was asked. That is a different claim, and it is the
one a student most needs to be true.
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
import ssvep as ssvep_script  # noqa: E402

from exg.io import load  # noqa: E402

DATASET = REPO / "examples" / "sample-data"
ALPHA = DATASET / "sub-01" / "eeg" / "sub-01_task-alpha_eeg.vhdr"
SSVEP = DATASET / "sub-01" / "eeg" / "sub-01_task-ssvep_eeg.vhdr"

pytestmark = pytest.mark.skipif(
    not ALPHA.is_file(), reason="sample data not present"
)


def test_the_alpha_recording_loads_with_its_events():
    recording = load(ALPHA)
    assert recording.fs == 1000.0
    assert recording.channels == ["O1", "O2"]
    assert len(recording.markers_named("eyes_closed")) == 3
    assert len(recording.markers_named("eyes_open")) == 3
    # Onsets were measured from a digital line, not assumed from the schedule.
    assert recording.markers_named("eyes_closed")[0].onset == pytest.approx(22.064, abs=0.01)


def test_the_alpha_recording_shows_the_berger_effect():
    """The result this recording exists to demonstrate.

    Alpha power should be several times larger with the eyes closed, and the
    peak should sit in the alpha band. Both are properties of the recording,
    so this test fails if the reader or the analysis ever stops working.
    """
    results = alpha_script.analyse(load(ALPHA), ["O1", "O2"])
    assert results["reactivity_ratio"] > 4.0
    assert results["reactivity_normalised"] > 0.6
    assert 8.0 <= results["peak_frequency_hz"] <= 13.0


def test_the_alpha_effect_is_present_on_each_channel_alone():
    recording = load(ALPHA)
    for site in ("O1", "O2"):
        results = alpha_script.analyse(recording, [site])
        assert results["reactivity_ratio"] > 3.0, site


def test_the_ssvep_recording_loads_with_its_stimulus_channel():
    recording = load(SSVEP)
    assert recording.fs == 3125.0
    assert recording.channels == ["O1", "O2", "Fpz"]
    # The stimulus line is kept, but out of the signal: it is a trigger, and
    # averaging a square wave into an occipital pair would manufacture a
    # response at exactly the frequency being looked for.
    assert "StimTrig" in recording.digital
    assert len(recording.markers_named("flicker17")) == 3


def test_the_stimulus_channel_reveals_the_true_flicker_rate():
    """The display did not run at exactly the rate it was asked for.

    A screen builds a flickering stimulus from whole frames, so it can only
    present its refresh rate divided by a whole number. 17 Hz does not divide
    into 60 Hz, so the display alternated between nearby rates and averaged
    out slightly below the target.

    This is worth asserting because it is what the analysis reports, and a
    reader might otherwise take the discrepancy for a bug.
    """
    from exg.io import find_rising_edges

    recording = load(SSVEP)
    onsets = find_rising_edges(recording.digital["StimTrig"], recording.fs, min_gap=0.005)
    intervals = np.diff(onsets)
    # Keep intervals within a stimulation block, dropping the gaps between them.
    within_block = intervals[intervals < 0.2]
    measured = 1.0 / np.mean(within_block)

    assert 16.0 < measured < 17.0
    assert measured == pytest.approx(16.72, abs=0.1)


def test_the_ssvep_response_appears_where_the_stimulus_actually_was():
    """The peak follows the display, not the number that was asked for."""
    results = ssvep_script.analyse(
        load(SSVEP), ["O1", "O2"], 17.0,
        stim_marker="flicker17", compute_plv=False,
    )
    # Within a quarter of a hertz of the true 16.72 Hz, not of the nominal 17.
    assert results["peak_frequency_hz"] == pytest.approx(16.72, abs=0.25)
    assert results["power_ratio"] > 2.0


def test_reading_the_dataset_description():
    from exg.bids import describe_bids_dataset

    description = describe_bids_dataset(DATASET)
    assert description["name"]
    assert len(description["recordings"]) == 2
    assert description["participants"][0]["participant_id"] == "sub-01"
