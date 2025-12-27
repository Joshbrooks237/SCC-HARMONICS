"""
Visual Spectrum Capture and Analysis Module
============================================

The first witness. What the eye can see.
RGB, polarized, UV, dermoscopy - we interrogate with every wavelength.

"The lesion presents itself. We observe with more than mortal sight."
"""

from .capture import MultiSpectrumVisualCapture, VisualCapture
from .features import VisualFeatureExtractor

__all__ = ["MultiSpectrumVisualCapture", "VisualCapture", "VisualFeatureExtractor"]

