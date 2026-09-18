# Copyright 2026 Nexstem India Private Limited (trading as Anthriq)
# Licensed under the Apache License, Version 2.0. See the LICENSE file.

"""
exg — the shared parts of the analysis scripts.

Three modules, kept here rather than in the scripts because all of the analyses
need them and three copies would drift apart:

    exg.io        reading recordings, and the markers that go with them
    exg.spectra   spectral estimation: PSD, band power, peaks, SNR, phase locking
    exg.plotting  figure styling

Everything else lives in the script that uses it. If you are reading the code to
learn how an analysis works, start in ``scripts/`` and come here for the
supporting maths.

The one contract worth knowing: **signals are in microvolts** once loaded, so
band powers come out in uV**2. The readers convert on the way in.
"""

__version__ = "1.0.0"
