"""
Combinatorial coverage optimizer for lottery portfolios.
v4.0 - tunable weights, empirical portfolio utilities.
"""
import random
import itertools
from collections import Counter
import sys
from pathlib import Path
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.config import (
    MIN_NUMBER, MAX_NUMBER, NUMBERS_PER_DRAW,
    COVERAGE_MONTE_CARLO_SAMPLES,
    DEFAULT_OPTIMIZER_WEIGHTS,
    DEFAULT_OPTIMIZER_CONSTRAINTS,
    RNG_SEED,
    logger
)


def _candidate_pairs(candidate):
    return set(itertools.combinations(candidate, 2))


def _candidate_triples(candidate):
    return set(itertools.combinations(candidate, 3))


def _ticket_overlap(a, b):
    return len(set(a) & set(b))


def _score_candidate(candidate, portfolio, covered_pairs, covered_triples,
                     weights, constraints):
    candidate_pairs = _candidate_pairs(candidate)
    candidate_triples = _candidate_triples(candidate)

    new_pairs = len(candidate_pairs - covered_pairs)
    new_triples = len(candidate_triples - covered_triples)

    overlap_penalty = 0.0
    threshold = constraints["overlap_penalty_threshold"]
    for existing in portfolio:
        overlap = _ticket_overlap(candidate, existing)
        if overlap >= threshold:
            overlap_penalty += (overlap - threshold + 1) * weights["w_overlap"]

    odd_count = sum(1 for n in candidate if n % 2 == 1)
    odd_penalty = 0.0
    if not (constraints["odd_min"] <= odd_count <= constraints["odd_max"]):
        odd_penalty = weights["w_odd_even_penalty"]

    ticket_sum = sum(candidate)
    expected_sum = NUMBERS_PER_DRAW * (MIN_NUMBER + MAX_NUMBER) / 2
    sum_deviation_ratio = abs(ticket_sum - expected_sum) / expected_sum
    sum_penalty = 0.0
    if sum_deviation_ratio > constraints["sum_tolerance_ratio"]:
        sum_penalty = weights["w_sum_penalty"]

    score = (
        weights["w_pairs"] * new_pairs +
        weights["w_triples"] * new_triples -
        overlap_penalty -
        odd_penalty -
        sum_penalty
    )

    diagnostics = {
        "new_pairs": new_pairs,
        "new_triples": new_triples,
        "overlap_penalty": overlap_penalty,
        "odd_penalty": odd_penalty,
        "sum_penalty": sum_penalty,
        "score": score,
    }
    return score, diagnostics


def optimize_portfolio_coverage(
    n_tickets,
    n_numbers=NUMBERS_PER_DRAW,
    min_num=MIN_NUMBER,
    max_num=MAX_NUMBER,
    monte_carlo_samples=None,
    weights=None,
    constraints=None,
    rng_seed=None,
):
    """
    Generate a low-overlap portfolio maximizing pair/triple diversity.
    """
    if monte_carlo_samples is None:
        monte_carlo_samples = COVERAGE_MONTE_CARLO_SAMPLES
    if weights is None:
        weights = DEFAULT_OPTIMIZER_WEIGHTS.copy()
    if constraints is None:
        constraints = DEFAULT_OPTIMIZER_CONSTRAINTS.copy()

    rng = random.Random(rng_seed if rng_seed is not None else RNG_SEED)
    all_numbers = list(range(min_num, max_num + 1))

    portfolio = []
    covered_pairs = set()
    covered_triples = set()

    total_possible_pairs = len(list(itertools.combinations(all_numbers, 2)))
    total_possible_triples = len(list(itertools.combinations(all_numbers, 3)))

    for ticket_idx in range(n_tickets):
        best_ticket = None
        best_score = float("-inf")

        for _ in range(monte_carlo_samples):
            candidate = sorted(rng.sample(all_numbers, n_numbers))
            score, _ = _score_candidate(
                candidate, portfolio, covered_pairs, covered_triples,
                weights, constraints
            )
            if score > best_score:
                best_score = score
                best_ticket = candidate

        if best_ticket is None:
            best_ticket = sorted(rng.sample(all_numbers, n_numbers))

        portfolio.append(best_ticket)
        covered_pairs.update(_candidate_pairs(best_ticket))
        covered_triples.update(_candidate_triples(best_ticket))

        logger.debug(
            f"Ticket {ticket_idx + 1}/{n_tickets}: {best_ticket} | "
            f"pair coverage={len(covered_pairs)}/{total_possible_pairs}"
        )

    stats = calculate_portfolio_statistics(
        portfolio=portfolio,
        covered_pairs=covered_pairs,
        covered_triples=covered_triples,
        total_possible_pairs=total_possible_pairs,
        total_possible_triples=total_possible_triples,
        weights=weights,
        constraints=constraints,
    )
    return portfolio, stats


def generate_random_portfolio(
    n_tickets,
    n_numbers=NUMBERS_PER_DRAW,
    min_num=MIN_NUMBER,
    max_num=MAX_NUMBER,
    rng_seed=None,
):
    rng = random.Random(rng_seed if rng_seed is not None else RNG_SEED)
    all_numbers = list(range(min_num, max_num + 1))
    portfolio = [sorted(rng.sample(all_numbers, n_numbers)) for _ in range(n_tickets)]

    covered_pairs = set()
    covered_triples = set()
    for ticket in portfolio:
        covered_pairs.update(_candidate_pairs(ticket))
        covered_triples.update(_candidate_triples(ticket))

    total_possible_pairs = len(list(itertools.combinations(all_numbers, 2)))
    total_possible_triples = len(list(itertools.combinations(all_numbers, 3)))

    stats = calculate_portfolio_statistics(
        portfolio=portfolio,
        covered_pairs=covered_pairs,
        covered_triples=covered_triples,
        total_possible_pairs=total_possible_pairs,
        total_possible_triples=total_possible_triples,
        weights=None,
        constraints=None,
    )
    return portfolio, stats


def calculate_portfolio_statistics(
    portfolio,
    covered_pairs=None,
    covered_triples=None,
    total_possible_pairs=None,
    total_possible_triples=None,
    weights=None,
    constraints=None,
):
    all_numbers_used = set()
    for ticket in portfolio:
        all_numbers_used.update(ticket)

    overlaps = []
    for i in range(len(portfolio)):
        for j in range(i + 1, len(portfolio)):
            overlaps.append(_ticket_overlap(portfolio[i], portfolio[j]))

    if covered_pairs is None:
        covered_pairs = set()
        for ticket in portfolio:
            covered_pairs.update(_candidate_pairs(ticket))

    if covered_triples is None:
        covered_triples = set()
        for ticket in portfolio:
            covered_triples.update(_candidate_triples(ticket))

    all_numbers = list(range(MIN_NUMBER, MAX_NUMBER + 1))
    if total_possible_pairs is None:
        total_possible_pairs = len(list(itertools.combinations(all_numbers, 2)))
    if total_possible_triples is None:
        total_possible_triples = len(list(itertools.combinations(all_numbers, 3)))

    number_freq = Counter()
    for ticket in portfolio:
        for n in ticket:
            number_freq[n] += 1

    freq_values = list(number_freq.values()) if number_freq else [0]
    most_common = number_freq.most_common(1)
    least_common = number_freq.most_common()[-1:] if number_freq else []

    diversity_score = (
        (len(covered_pairs) / total_possible_pairs) * 0.6 +
        (len(covered_triples) / total_possible_triples) * 0.4
        if total_possible_pairs > 0 and total_possible_triples > 0 else 0.0
    )

    return {
        "total_tickets": len(portfolio),
        "unique_numbers": len(all_numbers_used),
        "number_coverage_pct": len(all_numbers_used) / MAX_NUMBER * 100,
        "pairs_covered": len(covered_pairs),
        "pairs_total": total_possible_pairs,
        "pair_coverage_pct": len(covered_pairs) / total_possible_pairs * 100 if total_possible_pairs else 0.0,
        "triples_covered": len(covered_triples),
        "triples_total": total_possible_triples,
        "triple_coverage_pct": len(covered_triples) / total_possible_triples * 100 if total_possible_triples else 0.0,
        "avg_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "max_overlap": max(overlaps) if overlaps else 0,
        "min_overlap": min(overlaps) if overlaps else 0,
        "number_freq_std": float(np.std(freq_values)),
        "most_used_number": most_common[0] if most_common else (0, 0),
        "least_used_number": least_common[0] if least_common else (0, 0),
        "diversity_score": float(diversity_score),
        "weights": weights,
        "constraints": constraints,
    }


# Optional class wrapper for backward compatibility
class CoverageOptimizer:
    def __init__(self, rng_seed=None, weights=None, constraints=None, monte_carlo_samples=None):
        self.rng_seed = rng_seed
        self.weights = weights or DEFAULT_OPTIMIZER_WEIGHTS.copy()
        self.constraints = constraints or DEFAULT_OPTIMIZER_CONSTRAINTS.copy()
        self.monte_carlo_samples = monte_carlo_samples or COVERAGE_MONTE_CARLO_SAMPLES

    def generate_balanced_portfolio(self, n_tickets):
        portfolio, _ = optimize_portfolio_coverage(
            n_tickets=n_tickets,
            weights=self.weights,
            constraints=self.constraints,
            monte_carlo_samples=self.monte_carlo_samples,
            rng_seed=self.rng_seed,
        )
        return portfolio