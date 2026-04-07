"""
npaf_extraction.py — N-PAF extraction from any supported EEG format.

Port of nPAF.m. For each subject, loads a cleaned EEG recording via the
library's format-agnostic reader, computes an autoregressive (Yule-Walker)
power spectrum on posterior channels, and detects the normalised peak alpha
frequency (N-PAF) via sign-change of the log2-amplitude gradient in the
7–13 Hz band.

Supported formats: EDF, BDF, FIF, BrainVision, EEGLab, MATLAB, CSV, TSV.
Outputs per-subject CSV files and a summary across all subjects.
"""

import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.regression.linear_model import yule_walker

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================================
# CONSTANTS
# ============================================================================

POSTERIOR_CHANNELS = ["P3", "P4", "P7", "P8", "PZ", "O1", "O2", "OZ"]
MODEL_ORDER        = 256
FREQ_RESOLUTION    = 0.1          # Hz
ALPHA_BAND         = (7.0, 13.0)  # Hz
MIN_SEGMENT_SEC    = 2.0          # seconds
FFT_POINTS         = 2 ** 12

SUPPORTED_EXTENSIONS = {".edf", ".bdf", ".fif", ".vhdr", ".set", ".mat", ".csv", ".tsv"}


# ============================================================================
# CLASS
# ============================================================================

class NPAFExtractor:
    """
    Extract normalised peak alpha frequency (N-PAF) from any supported EEG recording.

    Uses the library's dataReader for format detection and loading, so any
    format accepted by eeg.utils.dataReader (EDF, BDF, FIF, BrainVision,
    EEGLab, MATLAB, CSV, TSV) works without additional configuration.

    Parameters
    ----------
    model_order : int
        Autoregressive model order for Yule-Walker estimation (default 256).
    freq_resolution : float
        Frequency resolution of the output spectrum in Hz (default 0.1).
    alpha_band : tuple of float
        Lower and upper bounds of the alpha band in Hz (default (7, 13)).
    min_segment_sec : float
        Minimum continuous segment length in seconds to include (default 2.0).
    posterior_channels : list of str
        Channel labels to use for PAF computation.
    """

    def __init__(
        self,
        model_order: int = MODEL_ORDER,
        freq_resolution: float = FREQ_RESOLUTION,
        alpha_band: tuple = ALPHA_BAND,
        min_segment_sec: float = MIN_SEGMENT_SEC,
        posterior_channels: list = None,
    ):
        self.model_order        = model_order
        self.freq_resolution    = freq_resolution
        self.alpha_band         = alpha_band
        self.min_segment_sec    = min_segment_sec
        self.posterior_channels = posterior_channels or POSTERIOR_CHANNELS
        self.freq_range         = np.arange(0.1, 45.0 + freq_resolution, freq_resolution)

    # ------------------------------------------------------------------ #
    #  Data loading                                                        #
    # ------------------------------------------------------------------ #

    def load(self, path: str) -> mne.io.BaseRaw:
        """
        Load an EEG file using the library's format-agnostic reader.

        Delegates to eeg.utils.dataReader.load(), which auto-detects the
        format from the file extension. Raises if the file contains pre-epoched
        data — N-PAF extraction requires continuous Raw data.
        """
        from ..utils.dataReader import load as _load
        raw, _ = _load(path)
        if isinstance(raw, mne.BaseEpochs):
            raise ValueError(
                f"{Path(path).name}: N-PAF extraction requires continuous (Raw) data. "
                "Pre-epoched files are not supported."
            )
        return raw

    # ------------------------------------------------------------------ #
    #  Channel selection                                                   #
    # ------------------------------------------------------------------ #

    def select_posterior_channels(self, raw: mne.io.BaseRaw) -> np.ndarray:
        """
        Return data array (n_channels, n_samples) for available posterior channels.

        Raises
        ------
        ValueError
            If none of the requested posterior channels are found.
        """
        available = [ch for ch in self.posterior_channels if ch in raw.ch_names]
        if not available:
            raise ValueError(
                f"None of the posterior channels {self.posterior_channels} "
                f"found in {raw.ch_names}"
            )
        self._selected_channels = available
        indices = [raw.ch_names.index(ch) for ch in available]
        return raw.get_data()[indices, :]  # (n_ch, n_samples)

    # ------------------------------------------------------------------ #
    #  Segment extraction (avoid boundary events)                          #
    # ------------------------------------------------------------------ #

    def extract_clean_segments(self, raw: mne.io.BaseRaw, data: np.ndarray) -> np.ndarray:
        """
        Concatenate clean data segments, skipping regions flagged as 'boundary'.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw object (used to read annotations and sample rate).
        data : np.ndarray, shape (n_channels, n_samples)
            Full data array already extracted from raw.

        Returns
        -------
        np.ndarray, shape (n_channels, n_clean_samples)
            Concatenated clean samples.
        """
        fs = raw.info["sfreq"]
        min_samples = int(self.min_segment_sec * fs)
        n_total = data.shape[1]

        boundary_samples = []
        for ann in raw.annotations:
            if "boundary" in ann["description"].lower():
                onset_sample = int(ann["onset"] * fs)
                boundary_samples.append(onset_sample)
        boundary_samples = sorted(boundary_samples)

        if not boundary_samples:
            return data

        starts = [0] + [b + 1 for b in boundary_samples]
        ends   = [b - 1 for b in boundary_samples] + [n_total - 1]

        segments = []
        for s, e in zip(starts, ends):
            if (e - s + 1) >= min_samples:
                segments.append(data[:, s : e + 1])

        if not segments:
            raise ValueError("No clean segments of sufficient length found after boundary removal.")

        return np.concatenate(segments, axis=1)

    # ------------------------------------------------------------------ #
    #  Autoregressive PSD                                                  #
    # ------------------------------------------------------------------ #

    def ar_psd(self, channel_data: np.ndarray, fs: float) -> np.ndarray:
        """
        Estimate PSD via Yule-Walker AR model, returned on self.freq_range.

        The AR transfer function is H(z) = sigma / A(z), giving
        PSD = sigma^2 / |A(e^{j*omega})|^2.  Matches MATLAB's aryule / pyulear.

        Parameters
        ----------
        channel_data : np.ndarray, shape (n_samples,)
        fs : float

        Returns
        -------
        np.ndarray, shape (len(self.freq_range),)
            Power spectral density on self.freq_range. NaN if estimation fails.
        """
        try:
            channel_data = channel_data - channel_data.mean()
            ar_coeffs, sigma = yule_walker(channel_data, order=self.model_order, method="mle")

            b_coeffs = [np.sqrt(sigma)]
            a_coeffs = np.concatenate([[1.0], -ar_coeffs])

            _, h = signal.freqz(b_coeffs, a_coeffs, worN=FFT_POINTS, fs=fs)
            freqs_uniform = np.linspace(0, fs / 2, FFT_POINTS)
            psd_uniform   = np.abs(h) ** 2

            return np.interp(self.freq_range, freqs_uniform, psd_uniform)

        except Exception:
            return np.full(len(self.freq_range), np.nan)

    # ------------------------------------------------------------------ #
    #  N-PAF detection                                                     #
    # ------------------------------------------------------------------ #

    def detect_npaf(self, log2_amplitude: np.ndarray) -> float:
        """
        Find N-PAF as the alpha-band frequency at the highest log2-amplitude peak.

        Peak detection: sign change of gradient from positive to negative
        (local maximum) within the alpha band.

        Parameters
        ----------
        log2_amplitude : np.ndarray
            Log2-amplitude spectrum over self.freq_range.

        Returns
        -------
        float
            N-PAF in Hz, or NaN if no peak is detected.
        """
        alpha_mask  = (self.freq_range >= self.alpha_band[0]) & (self.freq_range <= self.alpha_band[1])
        alpha_freqs = self.freq_range[alpha_mask]
        alpha_spec  = log2_amplitude[alpha_mask]

        if not np.all(np.isfinite(alpha_spec)):
            return np.nan

        grad = np.diff(alpha_spec)
        peak_idx = np.where((grad[:-1] > 0) & (grad[1:] < 0))[0] + 1

        if len(peak_idx) == 0:
            return np.nan

        best = peak_idx[np.argmax(alpha_spec[peak_idx])]
        return float(alpha_freqs[best])

    # ------------------------------------------------------------------ #
    #  Per-subject pipeline                                                #
    # ------------------------------------------------------------------ #

    def process_subject(self, path: str) -> dict:
        """
        Run the full N-PAF pipeline for one EEG file.

        Parameters
        ----------
        path : str
            Path to any supported EEG file (EDF, BDF, FIF, BrainVision,
            EEGLab, MATLAB, CSV, TSV).

        Returns
        -------
        dict with keys:
            subject, n_channels, n_valid, paf_{channel}, mean_paf,
            median_paf, std_paf, min_paf, max_paf
        """
        fpath   = Path(path)
        subject = fpath.stem

        print(f"\n{'='*48}")
        print(f"PROCESSING: {subject}")
        print(f"{'='*48}")

        raw      = self.load(path)
        fs       = raw.info["sfreq"]
        data     = self.select_posterior_channels(raw)
        channels = self._selected_channels

        print(f"  Channels : {channels}")
        print(f"  fs       : {fs:.1f} Hz")

        data_clean = self.extract_clean_segments(raw, data)
        print(f"  Clean samples : {data_clean.shape[1]} ({data_clean.shape[1]/fs:.1f} s)")

        n_ch       = data_clean.shape[0]
        paf_values = np.full(n_ch, np.nan)

        print("  Computing AR spectra …")
        for i in range(n_ch):
            psd           = self.ar_psd(data_clean[i], fs)
            amp           = np.sqrt(np.maximum(psd, 0.0))
            log2_amp      = np.log2(amp + 1e-12)
            paf_values[i] = self.detect_npaf(log2_amp)

        valid = paf_values[np.isfinite(paf_values)]
        print(f"  Valid channels: {len(valid)}/{n_ch}")

        result = {"subject": subject, "n_channels": n_ch, "n_valid": len(valid)}
        for ch, paf in zip(channels, paf_values):
            result[f"paf_{ch}"] = paf

        if len(valid) > 0:
            result.update({
                "mean_paf"  : float(np.mean(valid)),
                "median_paf": float(np.median(valid)),
                "std_paf"   : float(np.std(valid, ddof=1)),
                "min_paf"   : float(np.min(valid)),
                "max_paf"   : float(np.max(valid)),
            })
            print(f"  Mean N-PAF  : {result['mean_paf']:.2f} Hz")
            print(f"  Median N-PAF: {result['median_paf']:.2f} Hz")
        else:
            result.update({"mean_paf": np.nan, "median_paf": np.nan,
                           "std_paf": np.nan, "min_paf": np.nan, "max_paf": np.nan})

        return result

    # ------------------------------------------------------------------ #
    #  Multi-subject runner                                                #
    # ------------------------------------------------------------------ #

    def run_batch(
        self,
        data_dir: str,
        out_dir: str,
        file_pattern: str = "*",
    ) -> pd.DataFrame:
        """
        Process all supported EEG files in data_dir and save results.

        Globs data_dir for files matching file_pattern whose extension is in
        the supported set (EDF, BDF, FIF, BrainVision, EEGLab, MATLAB, CSV, TSV).
        Files are processed in alphabetical order.

        Parameters
        ----------
        data_dir : str
            Directory containing EEG files.
        out_dir : str
            Directory for per-subject and summary CSV output.
        file_pattern : str
            Glob pattern relative to data_dir (default "*" — all files).
            Examples: "subject*_cleaned.edf", "*.set", "sub-*_task-rest*.bdf"
        """
        data_dir = Path(data_dir)
        out_dir  = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            f for f in data_dir.glob(file_pattern)
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not files:
            print(f"[run_batch] No supported EEG files found in {data_dir} matching '{file_pattern}'")
            return pd.DataFrame()

        print(f"[run_batch] Found {len(files)} file(s) to process.")

        records = []
        for fpath in files:
            try:
                result = self.process_subject(str(fpath))
                records.append(result)
                pd.DataFrame([result]).to_csv(
                    out_dir / f"npaf_{fpath.stem}.csv", index=False
                )
            except Exception as exc:
                print(f"  [ERROR] {fpath.name}: {exc}")
                continue

        summary = pd.DataFrame(records)
        summary.to_csv(out_dir / "npaf_summary.csv", index=False)
        print(f"\nSummary saved → {out_dir / 'npaf_summary.csv'}")
        return summary


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="N-PAF extraction — supports EDF, BDF, FIF, BrainVision, EEGLab, MATLAB, CSV, TSV"
    )
    parser.add_argument("data_dir",   help="Directory containing EEG files")
    parser.add_argument("out_dir",    help="Output directory for results")
    parser.add_argument("--pattern",  default="*",
                        help="Glob pattern for file selection (default: '*')")
    parser.add_argument("--model_order", type=int, default=256,
                        help="AR model order (default 256)")
    args = parser.parse_args()

    extractor = NPAFExtractor(model_order=args.model_order)
    summary   = extractor.run_batch(args.data_dir, args.out_dir, args.pattern)
    if not summary.empty:
        print(summary[["subject", "n_valid", "mean_paf", "median_paf", "std_paf"]].to_string(index=False))
