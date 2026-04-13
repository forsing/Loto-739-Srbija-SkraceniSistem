"""
Portfolio generation models for Loto Serbia - v4.0
"""
import numpy as np
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.core.coverage_optimizer import (
    optimize_portfolio_coverage,
    generate_random_portfolio,
    calculate_portfolio_statistics,
)
from lotto_ai.config import DEFAULT_OPTIMIZER_WEIGHTS


def generate_adaptive_portfolio(
    features,
    n_tickets=10,
    use_adaptive=True,
    strategy="coverage_optimized",
    optimizer_weights=None,
    optimizer_constraints=None,
    monte_carlo_samples=None,
    rng_seed=None,
):
    """
    Generate a portfolio using selected strategy.
    'adaptive' here refers to portfolio construction strategy, not number prediction.
    """
    weights = optimizer_weights or DEFAULT_OPTIMIZER_WEIGHTS.copy()

    if strategy == "pure_random" or not use_adaptive:
        portfolio, stats = generate_random_portfolio(
            n_tickets=n_tickets,
            rng_seed=rng_seed,
        )
        metadata = {
            "strategy": "pure_random",
            "n_tickets": n_tickets,
            "coverage_stats": _serialize_stats(stats),
            "optimizer_weights": None,
            "optimizer_constraints": None,
        }
        return portfolio, metadata

    elif strategy == "coverage_optimized":
        portfolio, stats = optimize_portfolio_coverage(
            n_tickets=n_tickets,
            monte_carlo_samples=monte_carlo_samples,
            weights=weights,
            constraints=optimizer_constraints,
            rng_seed=rng_seed,
        )
        metadata = {
            "strategy": "coverage_optimized",
            "n_tickets": n_tickets,
            "coverage_stats": _serialize_stats(stats),
            "optimizer_weights": weights,
            "optimizer_constraints": optimizer_constraints,
        }
        return portfolio, metadata

    elif strategy == "hybrid":
        n_optimized = max(1, int(n_tickets * 0.7))
        n_random = n_tickets - n_optimized

        optimized, _ = optimize_portfolio_coverage(
            n_tickets=n_optimized,
            monte_carlo_samples=monte_carlo_samples,
            weights=weights,
            constraints=optimizer_constraints,
            rng_seed=rng_seed,
        )
        random_tickets, _ = generate_random_portfolio(
            n_tickets=n_random,
            rng_seed=None if rng_seed is None else rng_seed + 12345,
        )

        portfolio = optimized + random_tickets
        stats = calculate_portfolio_statistics(portfolio)

        metadata = {
            "strategy": "hybrid",
            "n_tickets": n_tickets,
            "n_optimized": n_optimized,
            "n_random": n_random,
            "coverage_stats": _serialize_stats(stats),
            "optimizer_weights": weights,
            "optimizer_constraints": optimizer_constraints,
        }
        return portfolio, metadata

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _serialize_stats(stats):
    serializable = {}
    for k, v in stats.items():
        if isinstance(v, (np.integer, np.int64)):
            serializable[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            serializable[k] = float(v)
        elif isinstance(v, tuple):
            serializable[k] = list(v)
        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    return serializable


def portfolio_statistics(portfolio):
    return calculate_portfolio_statistics(portfolio)