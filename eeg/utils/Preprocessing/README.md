# preprocessing

Standard EEG preprocessing pipeline: bandpass filtering, notch filtering, and event-based epoching. Operates on any MNE Raw or Epochs object returned by the reader.

---

## What it does

### 1. Bandpass filter

A zero-phase FIR filter (Hamming window) is applied to remove slow drifts and high-frequency noise:

- Default passband: **0.5 – 40 Hz**
- Removes DC drift and sub-Hz movement artefacts at the low end
- Removes high-frequency muscle artefacts and aliasing above 40 Hz
- Window: Hamming (good stopband attenuation, minimal ringing)

Both cutoffs are configurable (`l_freq`, `h_freq`).

### 2. Notch filter

A notch filter removes power line interference and its harmonics up to Nyquist. The line frequency is **never defaulted** — it is either passed explicitly or the user is prompted:

```
Power line frequency not set.
  [1] 50 Hz  (Europe, Asia, Africa, most of the world)
  [2] 60 Hz  (North America, parts of South America, Japan)
Enter choice [1/2] or type the frequency directly:
```

Harmonics (e.g. 100, 150 Hz for 50 Hz mains) are included automatically.

### 3. Event-based epoching

If the input is continuous Raw data **and** markers were detected by the reader (`metadata["has_markers"] == True`), the data is segmented into epochs:

- Default window: **−0.2 s to +0.8 s** around each event onset
- Default baseline correction: pre-stimulus interval (None to 0)
- Events are extracted from MNE annotations using `mne.events_from_annotations`
- Both window bounds and baseline are configurable

If no markers are present, the filtered Raw is returned unchanged.

---

## Output

Returns `mne.io.BaseRaw` or `mne.BaseEpochs` depending on whether epoching was triggered.

---

## Usage

```python
from eeg.utils.dataReader import load
from eeg.utils.preprocessing import preprocess

data, meta = load("recording.edf")

# Will prompt for line frequency if not provided
processed = preprocess(data, meta)

# Or pass it directly
processed = preprocess(data, meta, line_freq=50)

# Custom epoch window
processed = preprocess(data, meta, line_freq=50, epoch_tmin=-0.1, epoch_tmax=1.0)
```

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `l_freq` | `0.5` | Bandpass lower cutoff (Hz) |
| `h_freq` | `40.0` | Bandpass upper cutoff (Hz) |
| `line_freq` | `None` | Notch frequency (Hz). Prompted if not provided. |
| `epoch_tmin` | `-0.2` | Epoch start relative to event onset (s) |
| `epoch_tmax` | `0.8` | Epoch end relative to event onset (s) |
| `baseline` | `(None, 0.0)` | Baseline correction window |

---

## Design notes

- The notch frequency prompt exists by design. 50 Hz and 60 Hz produce very different filter outputs; silently defaulting to one would corrupt recordings from the other grid. Users must confirm.
- Epoching is only attempted on continuous data with confirmed markers. Pre-epoched files pass through the filter steps and are returned as-is.
- If event extraction fails (malformed annotations), the filtered Raw is returned with a warning rather than raising.
