"""
Statistical analysis models - for DISPLAY only, not prediction.
Honest about what frequency analysis can and cannot do.
"""
import pandas as pd
import numpy as np


def frequency_analysis_display(features):
    """
    Calculate empirical frequencies for display.
    NOT predictive - shown alongside disclaimer.
    """
    grouped = features.groupby("number")["hit"].agg(["sum", "count"])
    n_numbers = len(grouped)
    grouped["frequency"] = grouped["sum"] / grouped["count"]
    grouped["expected"] = 7.0 / 39.0
    grouped["deviation_pct"] = (
        (grouped["frequency"] - grouped["expected"]) / grouped["expected"] * 100
    )
    return grouped


def hot_cold_display(features, window=20):
    """
    Show which numbers are 'hot' (recent) vs 'cold' (overdue).
    FOR ENTERTAINMENT ONLY - these have no predictive power.
    """
    recent = features[features["draw_index"] >= features["draw_index"].max() - window]

    hot = recent.groupby("number")["hit"].sum().sort_values(ascending=False)
    cold = features.groupby("number")["gap"].last().sort_values(ascending=False)

    return {
        'hot_numbers': hot.head(10).index.tolist(),
        'cold_numbers': cold.head(10).index.tolist(),
        'disclaimer': (
            "Hot/cold numbers are DESCRIPTIVE only. "
            "A fair lottery has no memory - each draw is independent. "
            "Past frequency does NOT predict future results."
        )
    }