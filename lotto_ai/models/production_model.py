"""
Production model - redirects to core.models
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.core.models import (
    generate_optimized_portfolio,
    generate_adaptive_portfolio,
    frequency_analysis
)
from lotto_ai.core.coverage_optimizer import portfolio_statistics

__all__ = [
    'generate_optimized_portfolio',
    'generate_adaptive_portfolio',
    'portfolio_statistics',
    'frequency_analysis'
]