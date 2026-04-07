"""
anthriq-eeg — Open-source EEG processing algorithms from Anthriq.

Modules:
    EEG_Reader            — Format detection, data loading, metadata inference
    EEG_Preprocessing     — Bandpass/notch filtering and event-based epoching
    EEG_FeatureExtraction — Time-domain, frequency-domain, and nonlinear feature extraction
    cli                   — Command-line interface
"""

from .EEG_Reader import load
from .EEG_Preprocessing import preprocess
from .EEG_FeatureExtraction import extract_features

__all__ = ["load", "preprocess", "extract_features"]
__version__ = "0.1.0"
