# Anthriq Algorithms

Open-source biosignal analysis from Anthriq. Readable, runnable scripts for
analysing EXG recordings — EEG, ECG, EMG, EOG — whatever hardware produced
them.

They are written to be **read**, not just run. Each script is a few hundred
lines of commented NumPy and SciPy, and the comments explain the signal
processing rather than the syntax. If you want to know why a window length
matters, or what phase locking tells you that power does not, the answer is in
the file next to the code that does it.

## Quick start

No hardware needed. Generate a recording with a known answer and analyse it:

```bash
pip install numpy scipy matplotlib

python scripts/synth.py alpha /tmp/demo
python scripts/alpha.py /tmp/demo --sites O1,O2
```

The generator prints the ground truth it used. Compare it with what the
analysis reports — that is the only way to tell a working pipeline from a
broken one.

Real recordings are included too, so you can see what actual EEG looks like:

```bash
python scripts/alpha.py examples/sample-data/sub-01/eeg/sub-01_task-alpha_eeg.vhdr --sites O1,O2
```

## The analyses

| Script | Measures | Needs |
|---|---|---|
| `scripts/alpha.py` | Alpha reactivity: the rhythm that appears when you close your eyes | Two conditions, eyes closed and eyes open |
| `scripts/ssvep.py` | Steady-state response to a flickering stimulus | Flicker and rest blocks, and the flicker frequency |
| `scripts/emg.py` | Muscle activity: contraction strength and fatigue | A grip sequence, ideally with markers |
| `scripts/cmrr.py` | Common-mode rejection of the amplifier itself | A bench rig: signal generator, resistors, no subject |
| `scripts/synth.py` | Generates data with known answers for all four | Nothing |

Each takes `--help`.

### Alpha reactivity

```bash
python scripts/alpha.py my_recording/ --sites O1,O2
```

Reports band power in each condition, the reactivity ratio, a normalised index
bounded in ±1, and individual peak alpha frequency. A ratio above 2× is a
clear result.

### Steady-state visually evoked potential

```bash
python scripts/ssvep.py my_recording/ --stim-freq 12 --sites O1,O2 --control Fpz
```

Reports the peak frequency, signal-to-noise at the fundamental and its
harmonics, and the phase-locking value.

`--stim-freq` is required and is never guessed. A script that searched for the
strongest peak and then reported its signal-to-noise would look successful
whether or not a response existed — it would be measuring the largest thing in
the spectrum and calling it a response.

### Surface EMG

```bash
python scripts/emg.py my_recording/ --sites EMG1,EMG2
```

Reports an RMS envelope, per-contraction amplitude, and median frequency across
contractions. Amplitude alone cannot show fatigue, since it often rises as a
subject recruits harder; the spectral shift is what distinguishes the two, so
both are reported.

### Common-mode rejection

```bash
python scripts/cmrr.py captures/ --monitor ai0 --fs 2000
```

Takes a directory of captures, one per drive frequency, and reports the
rejection across the sweep. CMRR is a curve, not a number: anything imperfectly
matched between the amplifier's two input paths contributes an error that varies
with frequency, often by tens of decibels across the band that matters.

## Input formats

Three layouts are read, and the reader works out which is which from the
structure of what you point it at.

### BIDS

[BIDS](https://bids.neuroimaging.io) is the standard layout for shareable
neuroscience data. A great deal of public EEG is published this way, so you can
point these scripts at datasets nobody here recorded:

```bash
python scripts/alpha.py some_bids_dataset/ --sites O1,O2
python scripts/alpha.py some_bids_dataset/sub-01/eeg/sub-01_task-alpha_eeg.vhdr --sites O1,O2
```

The signal is read from BrainVision files, the channel types from
`*_channels.tsv`, and the conditions from `*_events.tsv`. Channels the dataset
marks as triggers or miscellaneous are kept separate rather than mixed into the
signal, so they cannot end up averaged into a region of interest by accident.

### BXI Studio export

Exporting a recording writes a folder:

```text
my_recording/
├── meta.json            sample rate, channel labels, units, and the markers
└── my_recording.csv     one row per sample
```

```text
O1,O2,Fpz,timestamp
12.5,-3.25,0.75,1718000000123456
```

Two things routinely catch people out:

- `timestamp` is in **microseconds**, as a large integer. Not seconds.
- **The markers are not in the CSV.** They are in `meta.json`. Analysing the
  CSV on its own loses every event.

### Raw DAQ capture

Recording straight from the acquisition device gives a plain CSV named after
the device's own terminals:

```text
Mod_9234/ai0,Mod_9234/ai1,Mod_9401/port0/line0
0.00123,-0.00047,0
```

- `ai*` columns are analogue, in **volts at the converter**. Pass `--gain` to
  recover the voltage at the electrode.
- `port*/line*` columns are digital marker lines carrying a **brief pulse** at
  each event, so events are found as rising edges rather than levels.
- There is **no timestamp column**, so `--fs` is required. There is no default:
  a wrong sampling rate silently rescales every frequency in every result.

## On MNE-Python, and what this repository is for

[MNE-Python](https://mne.tools) does everything here and a great deal more, with
a decade of validation behind it. If you are building something real, use it.

These scripts exist for a different reason: so you can *see* how the analyses
work. `welch_psd` in [exg/spectra.py](exg/spectra.py) is forty lines you can
read, modify and argue with. Every choice in it — the window, the overlap, the
detrending — is written down next to the code that makes it, along with why.
That is the whole offering, and it is why the dependencies stop at NumPy, SciPy
and Matplotlib: a reader you have to install a toolkit to use is just a worse
interface to the toolkit.

The natural path is to outgrow this. Read these, understand what the numbers
mean, then move to MNE knowing what it is doing on your behalf.

## Repository layout

```text
scripts/            The analyses. Read these.
├── alpha.py
├── ssvep.py
├── emg.py
├── cmrr.py
└── synth.py        Generates data with known answers
exg/                The shared parts, kept out of the scripts to avoid
├── io.py           duplicating them: readers and marker decoding,
├── bids.py         BIDS and BrainVision, spectral estimation,
├── spectra.py      figure styling
└── plotting.py
examples/
└── sample-data/    Two real recordings, in BIDS format
tests/              Ground-truth tests. Worth reading as worked examples.
eeg/                An earlier EEG feature-extraction package (see below)
```

`exg/` exists because the file readers and the spectral functions are shared by
every analysis, and four copies would drift apart. Everything else lives in the
script that uses it.

## Sample data

`examples/sample-data/` holds two real EEG recordings from one consenting
adult, published under CC0. They are in BIDS format, about 7 MB in total:

| Task | Channels | Rate | What is in it |
|---|---|---|---|
| `alpha` | O1, O2 | 1000 Hz | Eyes closed against eyes open, three times |
| `ssvep` | O1, O2, Fpz | 3125 Hz | A 17 Hz flicker, fifteen seconds at a time |

The SSVEP recording is worth a closer look than the alpha one. Its `StimTrig`
channel carries the flicker itself, one pulse per cycle, and measuring it shows
the display actually ran at **16.72 Hz** rather than the 17 Hz it was asked
for. A screen builds a flickering stimulus from whole frames, so it can only
present its refresh rate divided by a whole number, and 17 does not divide into
60. The analysis finds its peak at 16.72 Hz, which is the display being
honestly measured rather than the analysis being wrong.

That is a better thing to learn from than a clean result.

## Units

Signals are in **microvolts** everywhere once loaded. Band powers therefore come
out in µV², and a tone amplitude in µV. The loaders convert on the way in, and
each `Recording` records how, in `unit_note` — worth printing when a result
looks off by a power of ten.

## Tests

```bash
pip install pytest
python -m pytest tests/
```

The tests check results against answers known in closed form, not against
previous output. A sinusoid of peak amplitude *A* must return band power *A*²/2;
epochs with aligned phases must give a phase-locking value near 1 and scattered
ones near chance; a rejection sweep must come back identical at any gain. If you
change something in `exg/`, these will tell you whether you changed a result.

Two tests are worth reading on their own:

- `test_high_power_with_chance_phase_locking` — a rhythm at the stimulus
  frequency that is not driven by it. Power calls it a strong response; phase
  says nothing is locked. This is why both are measured.
- `test_cmrr_is_the_same_at_any_gain` — the executable form of the claim that
  the measurement's gain cancels.

## About `eeg/`

`eeg/` is an earlier package for EEG feature extraction and an N-PAF brain-aging
pipeline, built on MNE-Python. It is unrelated to the scripts above and needs
`mne` and `statsmodels`:

```bash
pip install mne statsmodels
python -m eeg.cli --help
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: your data and whatever
you build with these are yours, there is no contributor agreement to sign, and
the only thing the licence asks is that derived software carries the `NOTICE`
attribution.

## Licence

Apache License 2.0. Copyright 2026 Nexstem India Private Limited, trading as
Anthriq. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

You may use, modify and redistribute this, including commercially. If you
distribute a derivative work, section 4(d) of the licence requires you to carry
the `NOTICE` file's attribution with it.

If you use this in published work, please cite it — see
[CITATION.cff](CITATION.cff).
