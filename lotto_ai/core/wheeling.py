"""
Lottery wheeling system with mathematically provable guarantees.

A wheel generates tickets from chosen key numbers such that IF enough
of your keys appear in the draw, at least one ticket GUARANTEES
a minimum number of matches.
"""
import itertools
import random
import sys
from pathlib import Path
from math import comb

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.config import (
    MIN_NUMBER, MAX_NUMBER, NUMBERS_PER_DRAW, logger
)


def generate_full_wheel(key_numbers, n_per_ticket=NUMBERS_PER_DRAW):
    """
    Full wheeling: ALL combinations of key_numbers taken n_per_ticket at a time.

    Guarantee: If ALL drawn numbers are in your key set, you win the jackpot.

    Warning: Can produce enormous numbers of tickets!
    """
    key_numbers = sorted(key_numbers)

    if len(key_numbers) < n_per_ticket:
        raise ValueError(
            f"Need at least {n_per_ticket} key numbers, got {len(key_numbers)}"
        )

    n_tickets = comb(len(key_numbers), n_per_ticket)
    if n_tickets > 500:
        raise ValueError(
            f"Too many combinations ({n_tickets:,}). "
            f"Use abbreviated wheel or reduce key numbers to ≤15."
        )

    tickets = [
        sorted(list(combo))
        for combo in itertools.combinations(key_numbers, n_per_ticket)
    ]

    guarantee = {
        'type': 'full_wheel',
        'key_numbers': key_numbers,
        'n_key_numbers': len(key_numbers),
        'n_tickets': len(tickets),
        'guarantee': (
            f'If all {n_per_ticket} winning numbers are among your '
            f'{len(key_numbers)} key numbers, you WILL have the jackpot ticket.'
        ),
        'condition': f'All {n_per_ticket} drawn numbers must be in your key set',
        'verified': True,
        'coverage_pct': 100.0
    }

    return tickets, guarantee


def generate_abbreviated_wheel(key_numbers, guarantee_if_hit=3,
                                 guarantee_match=3,
                                 n_per_ticket=NUMBERS_PER_DRAW,
                                 max_tickets=50):
    """
    Abbreviated wheeling: minimal tickets guaranteeing coverage.

    IF `guarantee_if_hit` of your key_numbers appear in the draw,
    THEN at least one ticket has at least `guarantee_match` of those numbers.

    Args:
        key_numbers: Your chosen numbers (e.g., 12 numbers from 1-39)
        guarantee_if_hit: How many of your numbers must be drawn (e.g., 3)
        guarantee_match: Minimum matches guaranteed on a ticket (e.g., 3)
        n_per_ticket: Numbers per ticket (7)
        max_tickets: Maximum tickets to generate

    Returns:
        tickets, guarantee_info
    """
    key_numbers = sorted(key_numbers)
    n_keys = len(key_numbers)

    if n_keys < guarantee_if_hit:
        raise ValueError(
            f"Need at least {guarantee_if_hit} key numbers, got {n_keys}"
        )

    if guarantee_match > guarantee_if_hit:
        raise ValueError(
            f"guarantee_match ({guarantee_match}) cannot exceed "
            f"guarantee_if_hit ({guarantee_if_hit})"
        )

    if guarantee_match > n_per_ticket:
        raise ValueError(
            f"guarantee_match ({guarantee_match}) cannot exceed "
            f"numbers per ticket ({n_per_ticket})"
        )

    # All possible subsets of key_numbers of size guarantee_if_hit
    hit_subsets = list(itertools.combinations(key_numbers, guarantee_if_hit))
    total_subsets = len(hit_subsets)

    logger.info(f"Abbreviated wheel: {n_keys} numbers, "
                f"guarantee-{guarantee_if_hit}, {total_subsets} subsets to cover")

    uncovered = set(range(total_subsets))
    tickets = []

    # Numbers available to fill remaining positions
    other_numbers = [
        n for n in range(MIN_NUMBER, MAX_NUMBER + 1)
        if n not in key_numbers
    ]

    iteration = 0
    while uncovered and len(tickets) < max_tickets:
        iteration += 1
        best_ticket = None
        best_newly_covered = set()

        # How many key numbers to include in each candidate
        # Must include at least guarantee_match key numbers
        min_keys_in_ticket = min(guarantee_match, n_keys)
        max_keys_in_ticket = min(n_keys, n_per_ticket)

        n_candidates = min(3000, max(1000, total_subsets * 10))

        for _ in range(n_candidates):
            # Decide how many key numbers to put in this ticket
            n_from_keys = random.randint(min_keys_in_ticket, max_keys_in_ticket)
            n_fill = n_per_ticket - n_from_keys

            chosen_keys = sorted(random.sample(key_numbers, n_from_keys))

            if n_fill > 0 and other_numbers:
                fill_count = min(n_fill, len(other_numbers))
                chosen_fill = sorted(random.sample(other_numbers, fill_count))
                candidate = sorted(chosen_keys + chosen_fill)
            else:
                candidate = chosen_keys

            # Pad if needed (shouldn't happen normally)
            while len(candidate) < n_per_ticket:
                remaining = [
                    n for n in range(MIN_NUMBER, MAX_NUMBER + 1)
                    if n not in candidate
                ]
                if remaining:
                    candidate.append(random.choice(remaining))
                    candidate = sorted(candidate)
                else:
                    break

            if len(candidate) != n_per_ticket:
                continue

            # How many uncovered subsets does this ticket cover?
            newly_covered = set()
            ticket_set = set(candidate)

            for idx in uncovered:
                subset = set(hit_subsets[idx])
                matches_in_ticket = len(subset & ticket_set)
                if matches_in_ticket >= guarantee_match:
                    newly_covered.add(idx)

            if len(newly_covered) > len(best_newly_covered):
                best_newly_covered = newly_covered
                best_ticket = candidate

        if best_ticket is None or len(best_newly_covered) == 0:
            # Fallback: force-create a ticket covering at least one subset
            for idx in list(uncovered):
                subset_nums = list(hit_subsets[idx])
                # Include all numbers from this subset
                ticket_nums = list(subset_nums)
                # Fill remaining spots
                remaining_needed = n_per_ticket - len(ticket_nums)
                available = [
                    n for n in key_numbers + other_numbers
                    if n not in ticket_nums
                ]
                random.shuffle(available)
                ticket_nums.extend(available[:remaining_needed])
                ticket_nums = sorted(ticket_nums[:n_per_ticket])

                if len(ticket_nums) == n_per_ticket:
                    best_ticket = ticket_nums
                    # Recalculate coverage
                    best_newly_covered = set()
                    ticket_set = set(best_ticket)
                    for idx2 in uncovered:
                        subset = set(hit_subsets[idx2])
                        if len(subset & ticket_set) >= guarantee_match:
                            best_newly_covered.add(idx2)
                    break

        if best_ticket is not None:
            tickets.append(best_ticket)
            uncovered -= best_newly_covered
            logger.debug(
                f"Ticket {len(tickets)}: {best_ticket} | "
                f"Covered {len(best_newly_covered)} new subsets | "
                f"Remaining: {len(uncovered)}"
            )

    # Verify
    verified = verify_wheel_guarantee(
        tickets, key_numbers, guarantee_if_hit, guarantee_match
    )

    covered_count = total_subsets - len(uncovered)
    coverage_pct = covered_count / total_subsets * 100 if total_subsets > 0 else 100

    logger.info(f"Wheel complete: {len(tickets)} tickets, "
                f"{coverage_pct:.1f}% coverage of {guarantee_if_hit}-subsets")

    guarantee = {
        'type': 'abbreviated_wheel',
        'key_numbers': key_numbers,
        'n_key_numbers': n_keys,
        'n_tickets': len(tickets),
        'guarantee_if_hit': guarantee_if_hit,
        'guarantee_match': guarantee_match,
        'guarantee': (
            f'If {guarantee_if_hit} of your {n_keys} key numbers are drawn, '
            f'at least one ticket has {guarantee_match}+ of those numbers.'
        ),
        'subsets_to_cover': total_subsets,
        'subsets_covered': covered_count,
        'coverage_pct': coverage_pct,
        'verified': verified,
        'uncovered_remaining': len(uncovered)
    }

    if not verified:
        guarantee['warning'] = (
            f'Could not achieve full coverage with {max_tickets} max tickets. '
            f'{len(uncovered)} subsets uncovered. '
            f'Try increasing max_tickets or reducing key numbers.'
        )

    return tickets, guarantee


def verify_wheel_guarantee(tickets, key_numbers, guarantee_if_hit,
                            guarantee_match):
    """
    Exhaustively verify the wheeling guarantee.

    For EVERY possible subset of `guarantee_if_hit` numbers from key_numbers,
    check that at least one ticket contains `guarantee_match` of them.
    """
    for hit_combo in itertools.combinations(key_numbers, guarantee_if_hit):
        hit_set = set(hit_combo)
        covered = False
        for ticket in tickets:
            if len(hit_set & set(ticket)) >= guarantee_match:
                covered = True
                break
        if not covered:
            logger.debug(f"Uncovered subset: {hit_combo}")
            return False
    return True


def wheel_cost_estimate(n_key_numbers, guarantee_if_hit=3,
                         guarantee_match=3, n_per_ticket=NUMBERS_PER_DRAW):
    """
    Estimate tickets needed for a wheel.
    Uses covering design lower bounds.
    """
    n_subsets = comb(n_key_numbers, guarantee_if_hit)
    max_coverage_per_ticket = comb(n_per_ticket, guarantee_match)

    # Ceiling division for lower bound
    lower_bound = max(1, (n_subsets + max_coverage_per_ticket - 1)
                      // max_coverage_per_ticket)

    return {
        'n_key_numbers': n_key_numbers,
        'subsets_to_cover': n_subsets,
        'max_coverage_per_ticket': max_coverage_per_ticket,
        'estimated_min_tickets': lower_bound,
        'estimated_max_tickets': min(lower_bound * 4, n_subsets),
        'note': 'Actual count depends on greedy algorithm efficiency'
    }