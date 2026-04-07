# Brain_Aging_normalisedPeakAlphaFreq

N-PAF (Normalised Peak Alpha Frequency) pipeline for brain aging research. Covers two stages: per-subject biomarker extraction from raw EEG, and a full statistical analysis of N-PAF against age and cognitive reserve.

**Input:** Resting-state continuous (non-epoched) EEG in any supported format (EDF, BDF, FIF, BrainVision, EEGLab, MATLAB, CSV, TSV). **Output:** `npaf_extraction.py` produces N-PAF — a per-subject spectral biomarker (Hz) extracted from posterior channels. `npaf_age_analysis.py` takes N-PAF values alongside age and cognitive scores and predicts brain age, producing correlation statistics and a summary figure.

---

## Background

Peak Alpha Frequency (PAF) is the dominant oscillatory frequency in the alpha band (8–12 Hz) of the resting-state EEG. It is one of the most reproducible individual biomarkers in electrophysiology and has been studied for over 60 years. Its relevance to aging is well established:

- PAF **declines with age** at approximately 0.1–0.2 Hz per decade after midlife
- PAF correlates with **processing speed**, working memory, and general cognitive ability
- PAF is used as a proxy for **brain age** — individuals with a higher PAF for their chronological age show better cognitive reserve
- PAF is sensitive to neurodegeneration (Alzheimer's, mild cognitive impairment) and can shift years before clinical symptoms

The **Normalised PAF (N-PAF)** method improves on simple FFT-based peak detection by:
1. Using an **autoregressive (AR) model** (Yule-Walker estimation) rather than Welch's periodogram. AR spectra have higher effective frequency resolution on short segments and produce sharper, less noisy spectral peaks — critical when the alpha peak is broad or subtle.
2. Working in **log₂ amplitude** space rather than power, which compresses the dynamic range and makes peak boundaries cleaner.
3. Detecting peaks via **sign change of the gradient** of the log₂ amplitude spectrum, selecting the highest peak within the alpha band rather than the global spectrum maximum.

---

## Modules

### `npaf_extraction.py` — Per-subject N-PAF extraction

**Class:** `NPAFExtractor`

Processes one or many EEG recordings and extracts N-PAF per subject. Accepts any format supported by the library's `dataReader` — no format-specific configuration required.

**Supported formats:** EDF, BDF, FIF, BrainVision (`.vhdr`), EEGLab (`.set`), MATLAB (`.mat`), CSV, TSV.

#### Pipeline steps

| Step | What happens |
|---|---|
| 1. Load | File loaded via `eeg.utils.dataReader.load()` — format is auto-detected from extension. Pre-epoched files raise an error; N-PAF requires continuous Raw data. |
| 2. Channel selection | Posterior channels selected: P3, P4, P7, P8, Pz, O1, O2, Oz. Only channels present in the file are used. |
| 3. Segment extraction | Continuous data is split at `boundary` annotations. Segments shorter than `min_segment_sec` (default 2 s) are discarded. Remaining segments are concatenated. |
| 4. AR spectrum | Each channel is demeaned. Yule-Walker AR model (order 256) is fitted via `statsmodels`. The transfer function `H(z) = σ / A(z)` is evaluated using `scipy.signal.freqz` at 4096 points, then interpolated onto a 0.1 Hz grid from 0.1 to 45 Hz. |
| 5. Log₂ amplitude | `log₂(√PSD + ε)` is computed per channel. |
| 6. Peak detection | Within 7–13 Hz: gradient sign changes from positive to negative (local maxima) are located. The highest-amplitude peak is taken as N-PAF. Channels with no detectable peak return NaN. |
| 7. Aggregation | Mean, median, SD, min, max across valid channels are computed per subject. |
| 8. Output | Per-subject dict and optional CSV saved to disk. |

#### AR model order rationale

Order 256 is high relative to standard spectral estimation but is appropriate here because:
- Resting-state EEG at 256–1000 Hz sampling rates has long-range temporal correlations
- High model order allows the AR spectrum to resolve narrow peaks that a low-order model would smear
- The Yule-Walker estimator is stable at this order for typical EEG segment lengths (≥ 2 s × sampling rate samples)

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_order` | `256` | AR model order for Yule-Walker estimation |
| `freq_resolution` | `0.1` | Output spectrum frequency resolution (Hz) |
| `alpha_band` | `(7.0, 13.0)` | Alpha band search range (Hz) |
| `min_segment_sec` | `2.0` | Minimum clean segment length before discarding |
| `posterior_channels` | P3, P4, P7, P8, Pz, O1, O2, Oz | Channels to include |

#### Output

Per-subject result dict keys:

| Key | Description |
|---|---|
| `subject` | Filename stem |
| `n_channels` | Number of selected channels found in file |
| `n_valid` | Channels with a detectable alpha peak |
| `paf_{channel}` | Individual N-PAF per channel (NaN if no peak) |
| `mean_paf` | Mean N-PAF across valid channels (Hz) |
| `median_paf` | Median N-PAF (Hz) |
| `std_paf` | Standard deviation (Hz) |
| `min_paf`, `max_paf` | Range (Hz) |

#### Usage

```python
from eeg.Brain_Aging_normalisedPeakAlphaFreq import NPAFExtractor

extractor = NPAFExtractor()

# Single subject — any supported format
result = extractor.process_subject("subject01_cleaned.edf")
result = extractor.process_subject("subject01_cleaned.set")   # EEGLab still works
print(f"Mean N-PAF: {result['mean_paf']:.2f} Hz")
print(f"Valid channels: {result['n_valid']}/{result['n_channels']}")

# Full cohort — processes all supported EEG files in the directory
summary_df = extractor.run_batch(
    data_dir="./CleanData",
    out_dir="./Results",
)

# Narrow down with a glob pattern if needed
summary_df = extractor.run_batch(
    data_dir="./CleanData",
    out_dir="./Results",
    file_pattern="subject*_cleaned.edf",
)
```

CLI:

```bash
# Process all supported files in a directory
python -m eeg.Brain_Aging_normalisedPeakAlphaFreq.npaf_extraction \
    ./CleanData ./Results

# Filter by pattern
python -m eeg.Brain_Aging_normalisedPeakAlphaFreq.npaf_extraction \
    ./CleanData ./Results --pattern "subject*_cleaned.bdf" --model_order 256
```

---

### `npaf_age_analysis.py` — N-PAF vs Age / NART correlation analysis

**Function:** `run_npaf_age_analysis(age, nart, npaf, out_dir, figure_name)`

Implements a five-step statistical analysis relating N-PAF to chronological age and NART (National Adult Reading Test) scores.

#### What is NART?

The National Adult Reading Test estimates **premorbid IQ** from reading ability — it is largely resistant to acquired cognitive decline because irregular word pronunciation is learned early and well retained. In brain aging research, NART is used as a proxy for **cognitive reserve**: the brain's capacity to cope with age-related neurodegeneration before function is lost. The NART-N-PAF relationship, after controlling for age, tests whether N-PAF tracks cognitive reserve independently of the aging process itself.

#### Analysis steps

| Step | Analysis | Method |
|---|---|---|
| 1 | Age vs N-PAF correlation | Pearson r |
| 2 | Age vs NART correlation | Pearson r |
| 3 | N-PAF vs NART (simple) | Pearson r |
| 4 | **Partial correlation: N-PAF vs NART controlling for Age** | OLS residuals method: regress both N-PAF and NART onto Age separately, correlate residuals |
| 5 | Multiple regression: NART ~ Age + N-PAF | Normal equations (least squares) |
| 6 | Age-stratified analysis (median split) | Pearson r within young and old subgroups |

Step 4 is the primary analysis. It answers: *does N-PAF predict cognitive reserve over and above what age alone predicts?* The residuals method is equivalent to `partialcorr` in MATLAB and gives the same result as the beta coefficient in the multiple regression for standardised variables.

#### Output

- Returns a dict of all computed statistics (`r_age_npaf`, `p_age_npaf`, `r_age_nart`, `p_age_nart`, `r_npaf_nart`, `p_npaf_nart`, `r_partial`, `p_partial`, regression coefficients, stratified correlations)
- Saves a **4-panel figure**:
  - Panel 1: Age vs N-PAF scatter + regression line
  - Panel 2: Age vs NART scatter + regression line
  - Panel 3: N-PAF vs NART (simple) scatter + regression line
  - Panel 4: Age-controlled residuals scatter (partial correlation visualisation)

#### Built-in reference dataset

The module ships with a hardcoded reference dataset (n=58 complete cases, originally 60 subjects):

- **Age**: range 20–78 years, median 46 years
- **NART**: raw error scores
- **N-PAF**: extracted from resting-state eyes-closed EEG (this pipeline)

Reference results on this dataset:

| Comparison | r | p |
|---|---|---|
| Age vs N-PAF | −0.459 | 0.0003 |
| Age vs NART | 0.087 | 0.517 |
| N-PAF vs NART (simple) | −0.222 | 0.095 |
| N-PAF vs NART (age-controlled) | −0.205 | 0.122 |

#### Usage

```python
from eeg.Brain_Aging_normalisedPeakAlphaFreq import run_npaf_age_analysis
import numpy as np

# Own data
results = run_npaf_age_analysis(
    age=np.array([...]),
    nart=np.array([...]),
    npaf=np.array([...]),
    out_dir="./Results",
    figure_name="npaf_analysis.png",
)

# Built-in reference dataset
results = run_npaf_age_analysis()

print(f"Partial r (N-PAF vs NART | Age): {results['r_partial']:.3f}, p = {results['p_partial']:.4f}")
```

CLI (from CSV with columns `age`, `nart`, `npaf`):

```bash
python -m eeg.Brain_Aging_normalisedPeakAlphaFreq.npaf_age_analysis \
    --csv npaf_summary.csv --out_dir ./Results --figure npaf_analysis.png
```

---

## Dependencies

| Package | Used for |
|---|---|
| `mne>=1.6` | File loading (via dataReader), annotation handling |
| `numpy>=1.24` | Array operations, gradient computation |
| `scipy>=1.11` | `signal.freqz` for AR transfer function evaluation, `stats.pearsonr`, `stats.linregress` |
| `statsmodels>=0.14` | Yule-Walker AR estimation (`regression.linear_model.yule_walker`) |
| `pandas>=2.0` | Per-subject CSV output, batch summary DataFrame |
| `matplotlib>=3.7` | 4-panel summary figure |
