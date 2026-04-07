"""
anthriq_eeg.Brain_Aging_NormalisedPeakAlphaFreq — N-PAF pipeline.

Modules:
    npaf_extraction    — per-subject N-PAF extraction from EEGLab .set files
    npaf_age_analysis  — N-PAF vs Age / NART correlation analysis
"""

from .npaf_extraction import NPAFExtractor
from .npaf_age_analysis import run_npaf_age_analysis

__all__ = ["NPAFExtractor", "run_npaf_age_analysis"]
