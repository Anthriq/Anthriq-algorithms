"""
cli.py — Command-line interface for anthriq-eeg.

Commands:
  anthriq-eeg read       <file>   Load and summarise an EEG file.
  anthriq-eeg preprocess <file>   Filter (and epoch) an EEG file, save result.
  anthriq-eeg features   <file>   Extract features from an EEG file, save CSV.
  anthriq-eeg pipeline   <file>   Run all three steps end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import click

from .utils.dataReader import load
from .utils.preprocessing import preprocess
from .utils.featureExtraction import extract_features


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="anthriq-eeg")
def main() -> None:
    """anthriq-eeg — Open-source EEG processing by Anthriq."""


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_file_arg = click.argument("file", type=click.Path(exists=True, dir_okay=False))

_line_freq_opt = click.option(
    "--line-freq", type=float, default=None,
    help="Power line frequency in Hz (50 or 60). Prompted if not provided.",
)
_output_opt = click.option(
    "--output", "-o", type=click.Path(), default=None,
    help="Output file path. Defaults to <input>_<suffix>.<ext>.",
)
_tmin_opt = click.option("--tmin", type=float, default=-0.2,
                         help="Epoch start relative to event (s). Default: -0.2")
_tmax_opt = click.option("--tmax", type=float, default=0.8,
                         help="Epoch end relative to event (s). Default: 0.8")


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

@main.command()
@_file_arg
def read(file: str) -> None:
    """Load an EEG file and print a metadata summary."""
    load(file)


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

@main.command()
@_file_arg
@_line_freq_opt
@_tmin_opt
@_tmax_opt
@_output_opt
def preprocess_cmd(
    file: str,
    line_freq: float | None,
    tmin: float,
    tmax: float,
    output: str | None,
) -> None:
    """Filter (and optionally epoch) an EEG file. Saves result as FIF."""
    data, meta = load(file)
    processed = preprocess(
        data, meta,
        line_freq=line_freq,
        epoch_tmin=tmin,
        epoch_tmax=tmax,
    )
    out = _resolve_output(file, output, suffix="_preprocessed", ext=".fif")
    processed.save(str(out), overwrite=True, verbose=False)
    click.echo(f"Saved preprocessed data → {out}")


# Register under the natural name
main.add_command(preprocess_cmd, name="preprocess")


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

@main.command()
@_file_arg
@_output_opt
def features(file: str, output: str | None) -> None:
    """Extract features from an EEG file and save as CSV."""
    data, _ = load(file)
    df = extract_features(data)
    out = _resolve_output(file, output, suffix="_features", ext=".csv")
    df.to_csv(str(out), index=False)
    click.echo(f"Extracted {len(df.columns) - 2} features for {len(df)} (epoch × channel) rows.")
    click.echo(f"Saved features → {out}")


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

@main.command()
@_file_arg
@_line_freq_opt
@_tmin_opt
@_tmax_opt
@_output_opt
def pipeline(
    file: str,
    line_freq: float | None,
    tmin: float,
    tmax: float,
    output: str | None,
) -> None:
    """Run the full pipeline: read → preprocess → extract features. Saves CSV."""
    click.echo("=== Step 1/3: Reading ===")
    data, meta = load(file)

    click.echo("=== Step 2/3: Preprocessing ===")
    processed = preprocess(
        data, meta,
        line_freq=line_freq,
        epoch_tmin=tmin,
        epoch_tmax=tmax,
    )

    click.echo("=== Step 3/3: Feature extraction ===")
    df = extract_features(processed)

    out = _resolve_output(file, output, suffix="_pipeline_features", ext=".csv")
    df.to_csv(str(out), index=False)
    click.echo(f"\nDone. {len(df)} rows × {len(df.columns) - 2} features saved → {out}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_output(
    source: str,
    explicit: str | None,
    suffix: str,
    ext: str,
) -> Path:
    if explicit:
        return Path(explicit)
    src = Path(source)
    return src.parent / (src.stem + suffix + ext)
