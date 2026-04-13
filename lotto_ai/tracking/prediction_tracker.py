"""
Prediction tracking - v3.0
Backward-compatible wrapper around core.tracker

This module exists so that any code doing:
    from lotto_ai.tracking.prediction_tracker import PredictionTracker
still works. All real logic lives in lotto_ai.core.tracker.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.core.tracker import PredictionTracker, PlayedTicketsTracker

__all__ = ['PredictionTracker', 'PlayedTicketsTracker']