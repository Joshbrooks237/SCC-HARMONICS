"""
Acoustic/Ultrasound Spectrum Analysis Module
============================================

THE THIRD WITNESS. THE KEY TO EVERYTHING.

From 40 kHz to 50 MHz, we sweep the acoustic spectrum.
Each frequency a question. Each harmonic an answer.
The cancer speaks in distortion. We have learned to listen.

"Normal tissue hums in perfect fifths.
 Malignancy DISTORTS.
 The second harmonic rises. The third. The tissue has lost its virtue."

THIS IS WHERE THE MAGIC HAPPENS.
"""

# The hardware interface - how we speak to the tissue
from .ultrasound_capture import UltrasoundCapture, UltrasoundHardwareInterface

# The harmonic analyzer - how we interpret the response
# 2nd through 8th harmonics, THD, spectral slope, phase coherence
from .harmonic_analysis import UltrasoundHarmonicAnalyzer

__all__ = ["UltrasoundCapture", "UltrasoundHardwareInterface", "UltrasoundHarmonicAnalyzer"]

