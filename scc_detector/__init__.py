"""
SCC Multi-Spectrum Early Detection System
==========================================

A comprehensive, multi-modal screening system for early detection of 
Squamous Cell Carcinoma using visual, thermal, and acoustic sensing.

    "The operating theater is no place for the timid.
     Nor is the laboratory. Nor is the algorithm.
     Step forward, or step aside."
                                        - The Philosophy

PHILOSOPHY: USE EVERYTHING. MISS NOTHING.

Components:
    - Visual: RGB, polarized, UV, dermoscopy (400-700nm)
    - Thermal: Infrared metabolic signatures (8-14μm)
    - Acoustic: Harmonic analysis across 40kHz-50MHz
    - Temporal: Evolution tracking over time
    - Fusion: Multi-modal AI integration

The cancer hides in the gaps between modalities.
We have closed those gaps.
"""

__version__ = "0.1.0"
__author__ = "SCC Detection Team"

# ═══════════════════════════════════════════════════════════════════════════════
# THE INSTRUMENTS - Each module a blade in the surgical arsenal
# ═══════════════════════════════════════════════════════════════════════════════

# The visual eye - sees what light reveals
from .visual import MultiSpectrumVisualCapture, VisualFeatureExtractor

# The thermal eye - sees what heat betrays
from .thermal import ThermalImagingSystem

# The acoustic ear - hears what echoes confess
# THIS IS WHERE THE MAGIC HAPPENS
from .acoustic import UltrasoundHarmonicAnalyzer

# The temporal memory - remembers what time changes
from .temporal import TemporalChangeDetector

# The fusion mind - synthesizes what none alone could grasp
from .fusion import MultiModalFusionEngine

# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "MultiSpectrumVisualCapture",
    "VisualFeatureExtractor", 
    "ThermalImagingSystem",
    "UltrasoundHarmonicAnalyzer",
    "TemporalChangeDetector",
    "MultiModalFusionEngine",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 
# "I am not a monster. I am simply ahead of the curve."
#
# ═══════════════════════════════════════════════════════════════════════════════
