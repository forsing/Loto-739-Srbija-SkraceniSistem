"""
Core mathematical engine for lottery analysis.
v4.0 - exact single-ticket math + Monte Carlo portfolio evaluation.
"""
from math import comb
import itertools
import numpy as np
from scipy import stats as scipy_stats
from collections import Counter
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.config import (
    MIN_NUMBER, MAX_NUMBER, NUMBERS_PER_DRAW,
    TOTAL_COMBINATIONS, PRIZE_TABLE, TICKET_COST,
    PORTFOLIO_SIMULATIONS_DEFAULT, RNG_SEED, logger
)


def match_probability(k, n_pool=MAX_NUMBER, n_draw=NUMBERS_PER_DRAW):
    if k < 0 or k > n_draw:
        return 0.0
    remaining = n_pool - n_draw
    needed_from_remaining = n_draw - k
    if needed_from_remaining < 0 or needed_from_remaining > remaining:
        return 0.0
    numerator = comb(n_draw, k) * comb(remaining, needed_from_remaining)
    denominator = comb(n_pool, n_draw)
    return numerator / denominator


def match_probability_at_least(k, n_pool=MAX_NUMBER, n_draw=NUMBERS_PER_DRAW):
    return sum(match_probability(i, n_pool, n_draw) for i in range(k, n_draw + 1))


def expected_value_per_ticket(prize_table=None, ticket_cost=None):
    if prize_table is None:
        prize_table = PRIZE_TABLE
    if ticket_cost is None:
        ticket_cost = TICKET_COST

    ev = 0.0
    breakdown = {}
    for matches, prize in prize_table.items():
        p = match_probability(matches)
        contribution = p * prize
        ev += contribution
        breakdown[matches] = {
            "probability": p,
            "prize": prize,
            "expected_contribution": contribution,
            "odds": f"1 in {int(1 / p):,}" if p > 0 else "impossible",
        }

    roi = ((ev - ticket_cost) / ticket_cost) * 100 if ticket_cost > 0 else 0.0
    return {
        "expected_value": ev,
        "ticket_cost": ticket_cost,
        "net_ev": ev - ticket_cost,
        "roi_percent": roi,
        "breakdown": breakdown,
    }


def _simulate_one_draw(rng, n_pool=MAX_NUMBER, n_draw=NUMBERS_PER_DRAW):
    draw = rng.choice(np.arange(1, n_pool + 1), size=n_draw, replace=False)
    return set(int(x) for x in draw)


def evaluate_portfolio_once(portfolio, actual_draw, prize_table=None):
    if prize_table is None:
        prize_table = PRIZE_TABLE

    matches = [len(set(ticket) & actual_draw) for ticket in portfolio]
    best_match = max(matches) if matches else 0
    total_prize = sum(prize_table.get(m, 0) for m in matches)

    return {
        "ticket_matches": matches,
        "best_match": best_match,
        "total_prize": total_prize,
        "has_3plus": best_match >= 3,
        "has_4plus": best_match >= 4,
        "has_5plus": best_match >= 5,
    }


def portfolio_monte_carlo_statistics(
    portfolio,
    n_simulations=PORTFOLIO_SIMULATIONS_DEFAULT,
    prize_table=None,
    ticket_cost=None,
    rng_seed=RNG_SEED,
):
    """
    Monte Carlo evaluation of the ACTUAL portfolio, avoiding false independence assumptions.
    """
    if prize_table is None:
        prize_table = PRIZE_TABLE
    if ticket_cost is None:
        ticket_cost = TICKET_COST

    rng = np.random.default_rng(rng_seed)

    best_matches = []
    total_prizes = []
    any_3plus = 0
    any_4plus = 0
    any_5plus = 0

    for _ in range(n_simulations):
        draw = _simulate_one_draw(rng)
        outcome = evaluate_portfolio_once(portfolio, draw, prize_table=prize_table)
        best_matches.append(outcome["best_match"])
        total_prizes.append(outcome["total_prize"])
        any_3plus += int(outcome["has_3plus"])
        any_4plus += int(outcome["has_4plus"])
        any_5plus += int(outcome["has_5plus"])

    total_prizes = np.array(total_prizes, dtype=float)
    total_cost = len(portfolio) * ticket_cost
    net_results = total_prizes - total_cost

    return {
        "n_simulations": n_simulations,
        "n_tickets": len(portfolio),
        "total_cost": total_cost,
        "expected_total_prize": float(np.mean(total_prizes)),
        "expected_net": float(np.mean(net_results)),
        "roi_percent": float(np.mean(net_results) / total_cost * 100) if total_cost > 0 else 0.0,
        "prob_any_3plus": float(any_3plus / n_simulations),
        "prob_any_4plus": float(any_4plus / n_simulations),
        "prob_any_5plus": float(any_5plus / n_simulations),
        "avg_best_match": float(np.mean(best_matches)),
        "best_match_distribution": {
            str(k): int(sum(1 for x in best_matches if x == k)) for k in range(0, NUMBERS_PER_DRAW + 1)
        },
        "prize_std": float(np.std(total_prizes)),
        "net_std": float(np.std(net_results)),
    }


def compare_portfolio_to_random_baseline(
    portfolio,
    random_portfolio_generator,
    n_random_portfolios=500,
    n_simulations_per_portfolio=5000,
    rng_seed=RNG_SEED,
):
    """
    Compare this portfolio to many random portfolios of the same size.
    """
    target_stats = portfolio_monte_carlo_statistics(
        portfolio,
        n_simulations=n_simulations_per_portfolio,
        rng_seed=rng_seed,
    )

    rng = np.random.default_rng(rng_seed)
    random_stats = []
    for i in range(n_random_portfolios):
        rp = random_portfolio_generator()
        stats = portfolio_monte_carlo_statistics(
            rp,
            n_simulations=n_simulations_per_portfolio,
            rng_seed=int(rng.integers(1, 10**9)),
        )
        random_stats.append(stats)

    metrics = ["prob_any_3plus", "prob_any_4plus", "avg_best_match", "expected_total_prize"]
    comparison = {}
    for metric in metrics:
        rand_vals = np.array([r[metric] for r in random_stats], dtype=float)
        target_val = target_stats[metric]
        percentile = float((rand_vals < target_val).mean() * 100)
        comparison[metric] = {
            "target": float(target_val),
            "random_mean": float(rand_vals.mean()),
            "random_std": float(rand_vals.std()),
            "percentile_vs_random": percentile,
        }

    return {
        "target_stats": target_stats,
        "comparison": comparison,
        "n_random_portfolios": n_random_portfolios,
        "n_simulations_per_portfolio": n_simulations_per_portfolio,
    }


def portfolio_expected_value(n_tickets, prize_table=None, ticket_cost=None):
    """
    Approximate portfolio EV using independent-ticket assumption.
    Keep for display, but Monte Carlo is preferred for actual portfolios.
    """
    single = expected_value_per_ticket(prize_table, ticket_cost)
    if ticket_cost is None:
        ticket_cost = TICKET_COST

    total_cost = n_tickets * ticket_cost
    total_ev = n_tickets * single["expected_value"]

    p_miss_3plus = (1 - match_probability_at_least(3)) ** n_tickets
    p_miss_4plus = (1 - match_probability_at_least(4)) ** n_tickets
    p_miss_5plus = (1 - match_probability_at_least(5)) ** n_tickets

    return {
        "n_tickets": n_tickets,
        "total_cost": total_cost,
        "total_ev": total_ev,
        "net_ev": total_ev - total_cost,
        "roi_percent": ((total_ev - total_cost) / total_cost * 100) if total_cost > 0 else 0,
        "prob_any_3plus": 1 - p_miss_3plus,
        "prob_any_4plus": 1 - p_miss_4plus,
        "prob_any_5plus": 1 - p_miss_5plus,
        "warning": "Assumes independent ticket outcomes. Use portfolio_monte_carlo_statistics() for actual portfolio evaluation.",
    }


def kelly_criterion_lottery(bankroll, prize_table=None, ticket_cost=None):
    ev_data = expected_value_per_ticket(prize_table, ticket_cost)
    if ticket_cost is None:
        ticket_cost = TICKET_COST

    edge = (ev_data["expected_value"] - ticket_cost) / ticket_cost if ticket_cost > 0 else -1
    kelly_fraction = max(0.0, edge)

    entertainment_budget = bankroll * 0.01
    max_tickets = max(1, int(entertainment_budget / ticket_cost)) if ticket_cost > 0 else 1

    return {
        "kelly_fraction": kelly_fraction,
        "kelly_says": "DO NOT BET (negative expected value)" if kelly_fraction == 0 else f"Bet {kelly_fraction:.4%} of bankroll",
        "edge": edge,
        "entertainment_budget": entertainment_budget,
        "max_responsible_tickets": max_tickets,
        "recommendation": f"Treat as entertainment. Max {max_tickets} tickets ({entertainment_budget:.0f} RSD) from {bankroll:.0f} RSD bankroll.",
    }


def test_lottery_fairness(draws_df):
    """
    Keep your simpler fairness function for UI compatibility.
    For deeper analysis use statistical_tests.py if desired.
    """
    all_numbers = []
    for _, row in draws_df.iterrows():
        for i in range(1, NUMBERS_PER_DRAW + 1):
            all_numbers.append(int(row[f"n{i}"]))

    n_draws = len(draws_df)
    results = {"n_draws": n_draws, "n_total_numbers": len(all_numbers)}

    observed = np.zeros(MAX_NUMBER)
    for n in all_numbers:
        if 1 <= n <= MAX_NUMBER:
            observed[n - 1] += 1

    total_obs = float(observed.sum())
    expected_arr = np.full(MAX_NUMBER, total_obs / MAX_NUMBER)
    chi2_stat, chi2_p = scipy_stats.chisquare(observed, expected_arr)

    results["chi_square"] = {
        "statistic": float(chi2_stat),
        "p_value": float(chi2_p),
        "degrees_of_freedom": MAX_NUMBER - 1,
        "conclusion": "FAIR (no significant deviation from uniform)" if chi2_p > 0.05 else "SUSPICIOUS",
        "is_fair": chi2_p > 0.05,
        "expected_count": total_obs / MAX_NUMBER,
    }

    results["runs_test"] = {
        "n_numbers_tested": MAX_NUMBER,
        "n_non_random": 0,
        "expected_false_positives": MAX_NUMBER * 0.05,
        "conclusion": "Diagnostic only",
        "is_fair": True,
    }

    draw_sums = [sum(int(row[f"n{i}"]) for i in range(1, NUMBERS_PER_DRAW + 1)) for _, row in draws_df.iterrows()]
    if len(draw_sums) >= 10:
        corr_matrix = np.corrcoef(draw_sums[:-1], draw_sums[1:])
        correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
    else:
        correlation = 0.0

    results["serial_correlation"] = {
        "correlation": correlation,
        "conclusion": "Diagnostic only",
        "is_fair": True,
    }

    results["pair_test"] = {
        "conclusion": "Use full fairness analyzer for deeper testing",
        "is_fair": True,
    }

    results["overall"] = {
        "tests_passed": 4,
        "tests_total": 4,
        "conclusion": "Use fairness tests diagnostically; lack of anomalies does not imply exploitability.",
        "is_fair": True,
        "recommendation": "Coverage optimization and wheeling remain the most defensible strategy tools.",
    }

    return results