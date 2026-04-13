"""
Rigorous backtest for Loto Serbia portfolio strategies - v4.0
Compares coverage-optimized vs random using paired empirical evaluation.
"""
import numpy as np
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.core.db import get_session, Draw
from lotto_ai.core.coverage_optimizer import optimize_portfolio_coverage, generate_random_portfolio
from lotto_ai.core.math_engine import evaluate_portfolio_once
from lotto_ai.config import (
    PRIZE_TABLE, TICKET_COST, RNG_SEED,
    BACKTEST_RANDOM_BASELINE_PORTFOLIOS, BACKTEST_BOOTSTRAP_SAMPLES,
    DEFAULT_OPTIMIZER_WEIGHTS
)


def bootstrap_ci(values, n_boot=5000, alpha=0.05, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    values = np.asarray(values, dtype=float)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots.append(sample.mean())
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def permutation_test_paired(diffs, n_perm=5000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    diffs = np.asarray(diffs, dtype=float)
    observed = abs(diffs.mean())
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs), replace=True)
        perm_mean = abs((diffs * signs).mean())
        if perm_mean >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def main():
    print("=" * 80)
    print("LOTO SERBIA PORTFOLIO OPTIMIZER - RIGOROUS BACKTEST")
    print("=" * 80)

    session = get_session()
    draws = session.query(Draw).order_by(Draw.draw_date).all()
    session.close()

    if len(draws) < 100:
        print(f"Need at least 100 draws for a meaningful backtest, found {len(draws)}")
        return

    n_test_draws = min(300, len(draws))
    test_draws = draws[-n_test_draws:]
    n_tickets = 6

    coverage_best = []
    random_best = []
    coverage_prize = []
    random_prize = []

    for i, draw in enumerate(test_draws):
        actual = set(draw.get_numbers())

        cov_portfolio, _ = optimize_portfolio_coverage(
            n_tickets=n_tickets,
            weights=DEFAULT_OPTIMIZER_WEIGHTS,
            rng_seed=RNG_SEED + i,
        )
        rnd_portfolio, _ = generate_random_portfolio(
            n_tickets=n_tickets,
            rng_seed=RNG_SEED + 100000 + i,
        )

        cov_outcome = evaluate_portfolio_once(cov_portfolio, actual, prize_table=PRIZE_TABLE)
        rnd_outcome = evaluate_portfolio_once(rnd_portfolio, actual, prize_table=PRIZE_TABLE)

        coverage_best.append(cov_outcome["best_match"])
        random_best.append(rnd_outcome["best_match"])
        coverage_prize.append(cov_outcome["total_prize"])
        random_prize.append(rnd_outcome["total_prize"])

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{n_test_draws} draws...")

    coverage_best = np.array(coverage_best, dtype=float)
    random_best = np.array(random_best, dtype=float)
    coverage_prize = np.array(coverage_prize, dtype=float)
    random_prize = np.array(random_prize, dtype=float)

    diff_best = coverage_best - random_best
    diff_prize = coverage_prize - random_prize

    ci_best = bootstrap_ci(diff_best, n_boot=BACKTEST_BOOTSTRAP_SAMPLES, rng_seed=RNG_SEED)
    ci_prize = bootstrap_ci(diff_prize, n_boot=BACKTEST_BOOTSTRAP_SAMPLES, rng_seed=RNG_SEED + 1)

    p_best = permutation_test_paired(diff_best, n_perm=BACKTEST_BOOTSTRAP_SAMPLES, rng_seed=RNG_SEED)
    p_prize = permutation_test_paired(diff_prize, n_perm=BACKTEST_BOOTSTRAP_SAMPLES, rng_seed=RNG_SEED + 1)

    total_cost = n_test_draws * n_tickets * TICKET_COST

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\nCoverage-Optimized")
    print(f"Avg best match: {coverage_best.mean():.4f}")
    print(f"3+ hit rate: {(coverage_best >= 3).mean():.2%}")
    print(f"4+ hit rate: {(coverage_best >= 4).mean():.2%}")
    print(f"Total prize: {coverage_prize.sum():,.0f} RSD")
    print(f"ROI: {(coverage_prize.sum() - total_cost) / total_cost * 100:.2f}%")

    print("\nPure Random")
    print(f"Avg best match: {random_best.mean():.4f}")
    print(f"3+ hit rate: {(random_best >= 3).mean():.2%}")
    print(f"4+ hit rate: {(random_best >= 4).mean():.2%}")
    print(f"Total prize: {random_prize.sum():,.0f} RSD")
    print(f"ROI: {(random_prize.sum() - total_cost) / total_cost * 100:.2f}%")

    print("\nPaired Comparison")
    print(f"Mean best-match difference (cov - rand): {diff_best.mean():.4f}")
    print(f"95% bootstrap CI: [{ci_best[0]:.4f}, {ci_best[1]:.4f}]")
    print(f"Permutation p-value: {p_best:.4f}")

    print(f"\nMean prize difference (cov - rand): {diff_prize.mean():.2f} RSD")
    print(f"95% bootstrap CI: [{ci_prize[0]:.2f}, {ci_prize[1]:.2f}]")
    print(f"Permutation p-value: {p_prize:.4f}")

    print("\nInterpretation")
    if p_best < 0.05 and diff_best.mean() > 0:
        print("Coverage portfolios appear better than random on best-match metric.")
    else:
        print("No strong evidence that coverage portfolios beat random on best-match metric.")

    if p_prize < 0.05 and diff_prize.mean() > 0:
        print("Coverage portfolios appear better than random on prize metric.")
    else:
        print("No strong evidence that coverage portfolios beat random on prize metric.")

    print("\nImportant caveat:")
    print("This tests portfolio construction quality, not prediction of future winning numbers.")
    print("=" * 80)


if __name__ == "__main__":
    main()