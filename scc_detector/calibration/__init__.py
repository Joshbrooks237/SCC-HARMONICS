"""
Calibration and Phantom Testing Module
======================================

Trust nothing. Verify everything.

A surgeon who does not calibrate his instruments
calibrates only his arrogance.

"The phantom teaches us what truth looks like.
 Only then can we recognize the lie."
"""

from .phantoms import TissuePhantom, CalibrationSystem

__all__ = ["TissuePhantom", "CalibrationSystem"]

