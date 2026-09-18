"""
eeg.utils — shared machinery for the EEG analyses.

The submodule directories are CapWords (``dataReader``, ``Preprocessing``,
``FeatureExtraction``). Import them by their real names: a lower-cased import
such as ``eeg.utils.preprocessing`` resolves only on case-insensitive
filesystems (macOS, Windows) and raises ``ModuleNotFoundError`` on Linux.
"""

from .dataReader import load
from .FeatureExtraction import extract_features
from .Preprocessing import preprocess

__all__ = ["load", "preprocess", "extract_features"]
