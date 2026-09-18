"""
Tests for the readers.

Most of these exist because of a specific way the format can be misread. Each
one names the trap it guards, because a reader that silently mis-parses is worse
than one that fails: the analysis still produces numbers, and they look fine.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from exg.io import (
    decode_markers,
    find_rising_edges,
    load,
    load_bxi_export,
    load_daq_csv,
)

FS = 1000.0
N = 5000


def write_bxi(folder, *, channels=("O1", "O2"), fs=FS, n=N, unit="uV",
              markers=None, declare_channels=True, timestamps=True):
    """Write a minimal BXI export. Deliberately hand-rolled rather than reusing
    the synthetic generator, so an encoder bug cannot hide a decoder bug."""
    folder.mkdir(parents=True, exist_ok=True)
    name = folder.name

    data = np.tile(np.linspace(1.0, 2.0, n), (len(channels), 1))
    columns = [data[i] for i in range(len(channels))]
    header = list(channels)
    if timestamps:
        # Absolute microseconds, as BXI writes them.
        header.append("timestamp")
        columns.append(1_718_000_000_000_000 + np.arange(n) * (1e6 / fs))

    with open(folder / f"{name}.csv", "w") as handle:
        handle.write(",".join(header) + "\n")
        for row in range(n):
            handle.write(",".join(f"{c[row]:.6f}" for c in columns) + "\n")

    meta = {"recordingId": name, "sampleRate": fs, "numChannels": len(channels)}
    if declare_channels:
        meta["channels"] = [
            {"id": f"c{i}", "label": c, "unit": unit} for i, c in enumerate(channels)
        ]
    if markers:
        meta["markers"] = markers
    (folder / "meta.json").write_text(json.dumps(meta))
    return folder


# --------------------------------------------------------------------------
# Sample rate
# --------------------------------------------------------------------------

def test_microsecond_timestamps_give_the_right_rate(tmp_path):
    """The trap: dividing 1 by a microsecond interval.

    A 1 kHz recording steps by 1000 microseconds per sample. Treating that as
    seconds gives 1/1000 = 0.001 Hz, and rounding it gives zero.
    """
    folder = write_bxi(tmp_path / "rec", fs=1000.0)
    # Drop sampleRate so the rate must come from the timestamps alone.
    meta = json.loads((folder / "meta.json").read_text())
    del meta["sampleRate"]
    (folder / "meta.json").write_text(json.dumps(meta))

    recording = load_bxi_export(folder)
    assert recording.fs == pytest.approx(1000.0, rel=1e-6)


def test_declared_rate_wins_over_the_timestamps(tmp_path):
    folder = write_bxi(tmp_path / "rec", fs=500.0)
    meta = json.loads((folder / "meta.json").read_text())
    meta["sampleRate"] = 250.0  # deliberately inconsistent
    (folder / "meta.json").write_text(json.dumps(meta))

    with pytest.warns(UserWarning, match="timestamps imply"):
        recording = load_bxi_export(folder)
    assert recording.fs == 250.0


def test_a_bare_csv_without_a_rate_refuses_to_guess(tmp_path):
    """Silently defaulting a sample rate rescales every frequency in the result.

    There is no safe default, so the loader must refuse rather than pick one.
    """
    path = tmp_path / "bench.csv"
    path.write_text("ai0,ai1\n0.1,0.2\n0.3,0.4\n")
    with pytest.raises(ValueError, match="sampling rate is unknown"):
        load(path)


def test_dropped_samples_are_reported(tmp_path):
    folder = write_bxi(tmp_path / "rec")
    lines = (folder / "rec.csv").read_text().splitlines()
    # Remove a run of samples so a gap appears in the timestamps.
    del lines[100:110]
    (folder / "rec.csv").write_text("\n".join(lines) + "\n")

    with pytest.warns(UserWarning, match="gap"):
        load_bxi_export(folder)


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

def test_microvolts_pass_through_unchanged(tmp_path):
    folder = write_bxi(tmp_path / "rec", unit="uV")
    recording = load_bxi_export(folder)
    assert recording.data.max() == pytest.approx(2.0, rel=1e-6)


def test_volts_are_converted_to_microvolts(tmp_path):
    """The trap: mixing unit conventions.

    A file in volts loaded as microvolts is wrong by a factor of a million, and
    every downstream number is wrong by the same factor, so nothing looks
    obviously broken.
    """
    folder = write_bxi(tmp_path / "rec", unit="V")
    with pytest.warns(UserWarning, match="far above anything physiological"):
        recording = load_bxi_export(folder)
    assert recording.data.max() == pytest.approx(2.0e6, rel=1e-6)


def test_an_explicit_unit_overrides_the_file(tmp_path):
    folder = write_bxi(tmp_path / "rec", unit="uV")
    recording = load_bxi_export(folder, unit="mV")
    assert recording.data.max() == pytest.approx(2.0e3, rel=1e-6)


def test_an_unrecognised_unit_warns_and_assumes_microvolts(tmp_path):
    folder = write_bxi(tmp_path / "rec", unit="furlongs")
    with pytest.warns(UserWarning, match="Unrecognised unit"):
        recording = load_bxi_export(folder)
    assert recording.data.max() == pytest.approx(2.0, rel=1e-6)


# --------------------------------------------------------------------------
# Channels
# --------------------------------------------------------------------------

def test_a_channel_named_like_metadata_is_still_a_channel(tmp_path):
    """The trap: filtering channel columns by name.

    Channel names are typed by whoever set up the recording, so one may well be
    called "Event" or "Trigger" or "Label". Any name-based filter would drop it.
    Selecting everything except the trailing timestamp column cannot.
    """
    folder = write_bxi(tmp_path / "rec", channels=("Event", "Trigger", "Label"))
    recording = load_bxi_export(folder)
    assert recording.channels == ["Event", "Trigger", "Label"]


def test_empty_declared_channels_fall_back_to_the_csv_header(tmp_path):
    """meta.json may set numChannels while listing no channels."""
    folder = write_bxi(tmp_path / "rec", channels=("O1", "O2"), declare_channels=False)
    recording = load_bxi_export(folder)
    assert recording.channels == ["O1", "O2"]


def test_picking_a_missing_channel_lists_what_is_available(tmp_path):
    folder = write_bxi(tmp_path / "rec", channels=("O1", "O2"))
    recording = load_bxi_export(folder)
    with pytest.raises(KeyError, match="Fpz"):
        recording.pick(["O1", "Fpz"])


def test_picking_is_case_and_whitespace_insensitive(tmp_path):
    folder = write_bxi(tmp_path / "rec", channels=("O1", "O2"))
    recording = load_bxi_export(folder)
    assert recording.pick([" o1 "]).shape[0] == 1


def test_a_header_data_mismatch_is_caught(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("ai0,ai1\n0.1,0.2,0.3\n")
    with pytest.raises(ValueError, match="columns"):
        load_daq_csv(path, fs=FS)


# --------------------------------------------------------------------------
# Markers in the sidecar
# --------------------------------------------------------------------------

# Verbatim from the documented format, so the decoder is tested against the
# specification rather than against our own encoder.
REFERENCE_MARKERS = {
    "v": 1,
    "defs": [
        {"id": 0, "name": "stimulus", "kind": "event"},
        {"id": 1, "name": "rest", "kind": "epoch"},
    ],
    "seq": {
        "def": [0, 0, 1],
        "dt": [500000, 250000, 250000],
        "dur": [0, 0, 2000000],
        "data": {"2": {"repetition": 3}},
    },
    "timeBase": "us",
    "origin": "recordingStart",
}


def test_marker_times_are_a_running_sum_of_the_gaps():
    """The trap: reading dt as an absolute time.

    dt is the gap since the *previous* marker. The reference sequence
    [500000, 250000, 250000] microseconds therefore means 0.5, 0.75 and 1.0 s
    -- not 0.5, 0.25, 0.25. Read as absolute, every marker lands too early and
    they bunch toward the start of the recording.
    """
    markers = decode_markers({"markers": REFERENCE_MARKERS})
    assert [m.onset for m in markers] == pytest.approx([0.5, 0.75, 1.0])


def test_marker_durations_distinguish_events_from_epochs():
    markers = decode_markers({"markers": REFERENCE_MARKERS})
    assert [m.duration for m in markers] == pytest.approx([0.0, 0.0, 2.0])


def test_marker_names_come_from_the_definition_ids():
    markers = decode_markers({"markers": REFERENCE_MARKERS})
    assert [m.name for m in markers] == ["stimulus", "stimulus", "rest"]


def test_payloads_are_keyed_by_index_as_a_string():
    """The trap: indexing the payload map with an integer.

    ``data`` is a JSON object whose keys are the occurrence index written as a
    string. An integer lookup finds nothing, silently, for every payload.
    """
    markers = decode_markers({"markers": REFERENCE_MARKERS})
    assert markers[2].payload == {"repetition": 3}
    assert markers[0].payload is None
    assert markers[1].payload is None


def test_no_marker_block_is_not_an_error():
    assert decode_markers({"sampleRate": 250.0}) == []


def test_an_unknown_definition_id_is_skipped_with_a_warning():
    broken = json.loads(json.dumps(REFERENCE_MARKERS))
    broken["seq"]["def"] = [0, 99, 1]
    with pytest.warns(UserWarning, match="unknown definition"):
        markers = decode_markers({"markers": broken})
    assert len(markers) == 2


def test_millisecond_timebase_is_honoured():
    other = json.loads(json.dumps(REFERENCE_MARKERS))
    other["timeBase"] = "ms"
    markers = decode_markers({"markers": other})
    assert markers[0].onset == pytest.approx(500.0)


def test_a_foreign_origin_warns():
    other = json.loads(json.dumps(REFERENCE_MARKERS))
    other["origin"] = "wallClock"
    with pytest.warns(UserWarning, match="origin"):
        decode_markers({"markers": other})


# --------------------------------------------------------------------------
# Markers on a digital line
# --------------------------------------------------------------------------

def test_a_brief_pulse_is_one_event_not_one_per_sample():
    """The trap: testing the level instead of the edge.

    The marker lines pulse for a few samples at each event. ``line != 0`` finds
    every sample inside each pulse, so a 3-sample pulse becomes 3 events and a
    100-sample pulse becomes 100.
    """
    line = np.zeros(10_000)
    for onset in (1000, 3000, 5000):
        line[onset:onset + 3] = 1.0
    onsets = find_rising_edges(line, FS)
    assert len(onsets) == 3
    assert onsets == pytest.approx([1.0, 3.0, 5.0])


def test_ttl_levels_work_without_configuration():
    line = np.zeros(1000)
    line[500:520] = 5.0
    assert len(find_rising_edges(line, FS)) == 1


def test_a_line_that_never_goes_high_has_no_events():
    assert find_rising_edges(np.zeros(1000), FS).size == 0


def test_a_constant_high_line_has_no_events():
    """A line stuck high has no transitions, so it carries no events.

    Without the flat-line guard the midpoint threshold equals the line value
    and every sample reads as high.
    """
    assert find_rising_edges(np.ones(1000), FS).size == 0


def test_bounce_within_the_debounce_window_is_one_event():
    line = np.zeros(1000)
    line[500] = 1.0
    line[502] = 1.0  # a ringing second edge 2 ms later
    assert len(find_rising_edges(line, FS, min_gap=0.01)) == 1


# --------------------------------------------------------------------------
# Raw DAQ captures
# --------------------------------------------------------------------------

def write_daq(path, *, headers, n=2000, value=0.001):
    with open(path, "w") as handle:
        handle.write(",".join(headers) + "\n")
        for _ in range(n):
            handle.write(",".join(f"{value:.6f}" for _ in headers) + "\n")
    return path


@pytest.mark.parametrize("headers", [
    ("Mod_9234/ai0", "Mod_9234/ai1"),
    ("ai0", "ai1"),
])
def test_both_column_naming_styles_are_recognised(tmp_path, headers):
    path = write_daq(tmp_path / "c.csv", headers=headers)
    recording = load_daq_csv(path, fs=FS, expect_physiological=False)
    assert len(recording.channels) == 2


def test_digital_columns_are_separated_from_analogue(tmp_path):
    path = write_daq(tmp_path / "c.csv",
                     headers=("ai0", "ai1", "port0/line0", "port0/line1"))
    recording = load_daq_csv(path, fs=FS, expect_physiological=False)
    assert recording.channels == ["ai0", "ai1"]
    assert set(recording.digital) == {"port0/line0", "port0/line1"}


def test_gain_is_divided_out(tmp_path):
    path = write_daq(tmp_path / "c.csv", headers=("ai0",), value=0.002)
    recording = load_daq_csv(path, fs=FS, gain=2000.0, expect_physiological=False)
    # 0.002 V = 2000 uV, divided by a gain of 2000 gives 1 uV.
    assert recording.data.max() == pytest.approx(1.0, rel=1e-6)


def test_epoch_durations_turn_pulses_into_spans(tmp_path):
    n = 3000
    with open(tmp_path / "c.csv", "w") as handle:
        handle.write("ai0,port0/line0\n")
        for row in range(n):
            handle.write(f"0.001,{1.0 if 1000 <= row < 1003 else 0.0}\n")

    recording = load_daq_csv(
        tmp_path / "c.csv", fs=FS,
        epoch_durations={"port0/line0": 10.0},
        marker_names={"port0/line0": "stimulus"},
        expect_physiological=False,
    )
    assert len(recording.markers) == 1
    assert recording.markers[0].name == "stimulus"
    assert recording.markers[0].duration == pytest.approx(10.0)


def test_a_capture_with_no_analogue_columns_is_rejected(tmp_path):
    path = write_daq(tmp_path / "c.csv", headers=("port0/line0",))
    with pytest.raises(ValueError, match="No analogue columns"):
        load_daq_csv(path, fs=FS)


def test_bench_captures_can_skip_the_physiological_check(tmp_path):
    """A CMRR rig drives volts on purpose, so the check must be suppressible."""
    path = write_daq(tmp_path / "c.csv", headers=("ai0",), value=1.5)
    recording = load_daq_csv(path, fs=FS, expect_physiological=False)
    assert recording.data.max() > 1e6


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def test_a_folder_with_meta_json_loads_as_a_bxi_export(tmp_path):
    folder = write_bxi(tmp_path / "rec")
    assert load(folder).fs == FS


def test_a_csv_beside_meta_json_loads_as_a_bxi_export(tmp_path):
    folder = write_bxi(tmp_path / "rec")
    assert load(folder / "rec.csv").fs == FS


def test_a_folder_without_meta_json_says_so(tmp_path):
    folder = tmp_path / "loose"
    folder.mkdir()
    (folder / "a.csv").write_text("ai0\n0.1\n")
    with pytest.raises(FileNotFoundError, match="no meta.json"):
        load(folder)


def test_recording_duration_and_summary(tmp_path):
    folder = write_bxi(tmp_path / "rec", markers=REFERENCE_MARKERS)
    recording = load_bxi_export(folder)
    assert recording.duration == pytest.approx(N / FS)
    assert recording.marker_names() == ["stimulus", "rest"]
    assert len(recording.markers_named("stimulus")) == 2
    summary = recording.summary()
    assert "Markers" in summary and "stimulus x2" in summary
