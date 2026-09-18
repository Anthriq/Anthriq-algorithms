# Anthriq Algorithms

Open-source biosignal analysis from Anthriq. These are readable, runnable
scripts for analysing EXG recordings — EEG, ECG, EMG, EOG — starting with the
experiments an xBud teaching kit is built around.

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

## The analyses

| Script | Measures | Needs |
|---|---|---|
| `scripts/alpha.py` | Alpha reactivity: the rhythm that appears when you close your eyes | Two conditions, eyes closed and eyes open |
| `scripts/ssvep.py` | Steady-state response to a flickering stimulus | Flicker and rest blocks, and the flicker frequency |
| `scripts/cmrr.py` | Common-mode rejection of the amplifier itself | A bench rig: signal generator, resistors, no subject |
| `scripts/synth.py` | Generates data with known answers for all three | Nothing |

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

### Common-mode rejection

```bash
python scripts/cmrr.py captures/ --monitor ai0 --fs 2000
```

Takes a directory of captures, one per drive frequency, and reports the
rejection across the sweep. CMRR is a curve, not a number: anything imperfectly
matched between the amplifier's two input paths contributes an error that varies
with frequency, often by tens of decibels across the band that matters.

## Input formats

Two layouts are read, and the reader works out which is which from the
structure of what you point it at.

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

## Repository layout

```text
scripts/            The analyses. Read these.
├── alpha.py
├── ssvep.py
├── cmrr.py
└── synth.py        Generates data with known answers
exg/                The shared parts, kept out of the scripts to avoid
├── io.py           triplicating them: readers and marker decoding,
├── spectra.py      spectral estimation, figure styling
└── plotting.py
tests/              Ground-truth tests. Worth reading as worked examples.
eeg/                An earlier EEG feature-extraction package (see below)
```

`exg/` exists because the file readers and the spectral functions are shared by
all three analyses, and three copies would drift apart. Everything else lives
in the script that uses it.

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

## Licence

Apache License 2.0. Copyright 2026 Anthriq. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).

You may use, modify and redistribute this, including commercially. If you
distribute a derivative work, section 4(d) of the licence requires you to carry
the `NOTICE` file's attribution with it.

If you use this in published work, please cite it — see
[CITATION.cff](CITATION.cff).
