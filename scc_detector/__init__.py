"""
SCC Multi-Spectrum Early Detection System

A comprehensive, multi-modal screening system for early detection of 
Squamous Cell Carcinoma using visual, thermal, and acoustic sensing.
"""

__version__ = "0.1.0"
__author__ = "SCC Detection Team"

from .visual import MultiSpectrumVisualCapture, VisualFeatureExtractor
from .thermal import ThermalImagingSystem
from .acoustic import UltrasoundHarmonicAnalyzer
from .temporal import TemporalChangeDetector
from .fusion import MultiModalFusionEngine

__all__ = [
    "MultiSpectrumVisualCapture",
    "VisualFeatureExtractor", 
    "ThermalImagingSystem",
    "UltrasoundHarmonicAnalyzer",
    "TemporalChangeDetector",
    "MultiModalFusionEngine",
]

