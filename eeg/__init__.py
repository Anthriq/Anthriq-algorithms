"""
eeg — Open-source EEG processing algorithms from Anthriq.

Modules:
    utils.dataReader        — Format detection, data loading, metadata inference
    utils.preprocessing     — Bandpass/notch filtering and event-based epoching
    utils.featureExtraction — Time-domain, frequency-domain, and nonlinear feature extraction
    cli                     — Command-line interface
"""

from .utils.dataReader import load
from .utils.preprocessing import preprocess
from .utils.featureExtraction import extract_features

__all__ = ["load", "preprocess", "extract_features"]
__version__ = "0.1.0"
