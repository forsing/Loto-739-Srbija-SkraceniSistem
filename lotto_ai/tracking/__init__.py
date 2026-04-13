"""
Tracking package - v3.0
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.tracking.prediction_tracker import PredictionTracker, PlayedTicketsTracker

__all__ = ['PredictionTracker', 'PlayedTicketsTracker']