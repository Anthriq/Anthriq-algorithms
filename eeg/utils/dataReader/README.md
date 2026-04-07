# dataReader

Format-agnostic EEG data loader. Point it at any supported EEG file and it returns a clean MNE object and a metadata dictionary — no configuration required.

---

## What it does

### 1. Format detection and loading

The format is inferred from the file extension. Each format has a dedicated MNE loader:

| Format | Extension | Loader |
|---|---|---|
| EDF | `.edf` | `mne.io.read_raw_edf` |
| BDF | `.bdf` | `mne.io.read_raw_bdf` |
| MNE FIF | `.fif` | `mne.io.read_raw_fif` / `mne.read_epochs` |
| BrainVision | `.vhdr` | `mne.io.read_raw_brainvision` |
| EEGLab | `.set` | `mne.io.read_raw_eeglab` / `mne.read_epochs_eeglab` |
| MATLAB | `.mat` | Custom loader — tries common variable names (`eeg`, `data`, `signal`, `x`) |
| CSV / TSV | `.csv`, `.tsv` | Custom loader — drops non-numeric columns, infers sample rate from time column |

### 2. Metadata inference

After loading, the reader builds a metadata dictionary automatically:

| Field | How it is inferred |
|---|---|
| `format` | File extension |
| `data_type` | `continuous` or `epoched` (from MNE object type) |
| `n_channels`, `channel_names`, `sfreq`, `duration_s` | From `mne.Info` |
| `has_markers`, `marker_description` | Annotations and event extraction via MNE |
| `task_type` | `task` / `rest` / `unknown` — keyword scan of README/BIDS files, then filename fallback |
| `n_subjects`, `n_sessions` | Parsed from `README.md`, `dataset_description.json`, or `participants.tsv` |

### 3. Unit inference

EEG amplitude unit is guessed from the **median absolute signal value** across a 10-second sample:

| Median amplitude | Likely unit |
|---|---|
| 1×10⁻⁷ – 1×10⁻³ V | µV-scale stored in Volts (MNE standard) |
| 1×10⁻⁴ – 5×10⁻⁴ V | Likely already in µV |
| > 5×10⁻⁴ V | Possibly mV or raw ADC counts |

The inferred unit and its confidence level are printed to the console and surfaced in the metadata. **Always verify manually** before proceeding with preprocessing or feature extraction.

---

## Output

```python
data, metadata = load("my_eeg.edf")
```

- `data` — `mne.io.BaseRaw` (continuous) or `mne.BaseEpochs` (pre-epoched)
- `metadata` — `dict` with all fields above

A formatted summary is printed to the console on every call.

---

## Usage

```python
from eeg.utils.dataReader import load

data, meta = load("recording.edf")

# Inspect
print(meta["format"])        # 'edf'
print(meta["data_type"])     # 'continuous'
print(meta["sfreq"])         # e.g. 512.0
print(meta["has_markers"])   # True / False
print(meta["unit"])          # inferred unit string
```

---

## Design notes

- All loaders use `preload=True` so data is in memory and immediately accessible.
- FIF and EEGLab formats are tried first as Raw, then as Epochs — whichever succeeds.
- For CSV/TSV: columns matching `time`, `marker`, `trigger`, `event`, `label`, `stim` are excluded from EEG channels. A marker column with non-zero values is attached as MNE annotations.
- For MATLAB `.mat`: sample rate is read from `srate`, `fs`, `Fs`, or `sfreq` (in that order), defaulting to 256 Hz if none are found.
- Metadata scanning walks both the file's directory and one level up (handles subject subfolders in BIDS-style datasets).
