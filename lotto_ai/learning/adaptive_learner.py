"""
Adaptive learner wrapper - v4.0
This adapts portfolio strategy mix, not lottery number prediction.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.core.learner import AdaptiveLearner

__all__ = ["AdaptiveLearner"]