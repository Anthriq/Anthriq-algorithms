"""
eeg — Open-source EEG processing algorithms from Anthriq.

Modules:
    utils.dataReader        — Format detection, data loading, metadata inference
    utils.Preprocessing     — Bandpass/notch filtering and event-based epoching
    utils.FeatureExtraction — Time-domain, frequency-domain, and nonlinear feature extraction
    cli                     — Command-line interface
"""

from .utils.dataReader import load
from .utils.FeatureExtraction import extract_features
from .utils.Preprocessing import preprocess

__all__ = ["load", "preprocess", "extract_features"]
__version__ = "0.1.0"
