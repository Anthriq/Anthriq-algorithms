# EEG_FeatureExtraction

Extracts 24 features per epoch per channel from any preprocessed EEG recording. Returns a flat pandas DataFrame ready for modelling, statistical analysis, or export.

---

## Output format

```
DataFrame shape: (n_epochs × n_channels) rows × 26 columns
Columns: epoch, channel, <24 feature columns>
```

For continuous Raw input, the entire recording is treated as a single pseudo-epoch (row label: `"continuous"`).

---

## Feature set

### Time-domain (12 features)

| Feature | Description |
|---|---|
| `hjorth_activity` | Signal variance — a measure of signal power |
| `hjorth_mobility` | Square root of variance of first derivative over signal variance — relates to mean frequency |
| `hjorth_complexity` | Ratio of mobility of second derivative to mobility of first — measures waveform complexity |
| `rms` | Root mean square amplitude |
| `mav` | Mean absolute value |
| `waveform_length` | Cumulative sum of absolute differences between successive samples — arc length of the signal |
| `zero_crossings` | Count of sign changes — a proxy for dominant frequency |
| `variance` | Signal variance (ddof=1) |
| `skewness` | Third standardised moment — asymmetry of the amplitude distribution |
| `kurtosis` | Fourth standardised moment — peakedness of the amplitude distribution |
| `peak_to_peak` | Maximum minus minimum amplitude |
| `mean_absolute_deviation` | Mean absolute deviation from the signal mean |

### Frequency-domain (9 features)

Computed via Welch's method (`scipy.signal.welch`, `nperseg=256`):

| Feature | Description |
|---|---|
| `total_power` | Total power across the full spectrum (area under PSD) |
| `median_frequency` | Frequency below which 50% of total power lies |
| `mean_frequency` | Power-weighted mean frequency |
| `band_power_delta` | Band power 1–4 Hz |
| `band_power_theta` | Band power 4–8 Hz |
| `band_power_alpha` | Band power 8–13 Hz |
| `band_power_beta` | Band power 13–30 Hz |
| `band_power_gamma` | Band power 30–40 Hz |
| `spectral_entropy` | Normalised Shannon entropy of the Welch PSD — high entropy = flat/complex spectrum, low entropy = narrow/dominant peak |

Band powers are computed as the area under the PSD curve within each band using `numpy.trapezoid`.

### Nonlinear (2 features)

| Feature | Description |
|---|---|
| `sample_entropy` | SampEn — measures unpredictability; lower values indicate more self-similar, regular signals. Template length m=2, tolerance r=0.2×std |
| `approximate_entropy` | ApEn — similar to SampEn but counts self-matches; faster but biased on short signals. Same parameters. |

---

## Usage

```python
from anthriq_eeg.EEG_Reader import load
from anthriq_eeg.EEG_Preprocessing import preprocess
from anthriq_eeg.EEG_FeatureExtraction import extract_features

data, meta = load("recording.edf")
processed = preprocess(data, meta, line_freq=50)
df = extract_features(processed)

print(df.shape)       # e.g. (320, 26) for 40 epochs × 8 channels
print(df.columns.tolist())
print(df.head())
```

Sampling frequency is read from `data.info["sfreq"]` automatically. Pass `sfreq` explicitly only if you are operating on a bare NumPy array outside MNE.

---

## Design notes

- Features are computed independently per epoch per channel — no cross-channel operations.
- The output DataFrame has features sorted alphabetically after `epoch` and `channel`, making column indexing predictable.
- Sample Entropy and Approximate Entropy are O(N²) — they will be slow on long continuous recordings. For resting-state recordings of several minutes, epoching first is strongly recommended.
- Spectral entropy uses log₂ and is normalised by the full PSD sum, giving values in bits. A flat white-noise spectrum produces maximum entropy; a single sharp peak produces near-zero entropy.
