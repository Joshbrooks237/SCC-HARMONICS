"""
Web UI for SCC Detection System
================================

The interface for those who prefer their surgery with a GUI.
We do not judge. We only serve.

"Medicine has always required both
 the precision of science
 and the artistry of presentation.
 
 The web is merely another operating theater."
"""

from .app import create_app, run_app

__all__ = ["create_app", "run_app"]

