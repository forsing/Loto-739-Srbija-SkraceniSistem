"""
Bankroll management for responsible lottery play.
Uses Kelly Criterion adaptation and responsible gambling limits.
"""
from math import comb
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.config import (
    logger, MATCH_PROBABILITIES, PRIZE_TABLE, TICKET_COST,
    EXPECTED_VALUE_PER_TICKET, NUMBERS_PER_DRAW, MAX_NUMBER,
    TOTAL_COMBINATIONS
)


class BankrollManager:
    """
    Provides mathematically-grounded spending advice.
    Key truth: lottery has negative expected value.
    """

    def __init__(self, ticket_cost=None, prize_table=None):
        self.ticket_cost = ticket_cost or TICKET_COST
        self.prize_table = prize_table or PRIZE_TABLE

    def calculate_expected_value(self):
        """
        Calculate exact expected value per ticket.
        """
        ev = 0
        breakdown = {}

        for k in range(NUMBERS_PER_DRAW + 1):
            prob = MATCH_PROBABILITIES[k]
            prize = self.prize_table.get(k, 0)
            ev_contribution = prob * prize

            breakdown[k] = {
                'matches': k,
                'probability': prob,
                'probability_1_in': 1 / prob if prob > 0 else float('inf'),
                'prize': prize,
                'ev_contribution': ev_contribution
            }
            ev += ev_contribution

        roi = (ev - self.ticket_cost) / self.ticket_cost * 100

        return {
            'expected_value': ev,
            'ticket_cost': self.ticket_cost,
            'net_expected_value': ev - self.ticket_cost,
            'roi_pct': roi,
            'house_edge_pct': -roi,
            'breakdown': breakdown,
            'interpretation': (
                f"For every {self.ticket_cost} RSD spent, you expect to get back "
                f"{ev:.2f} RSD on average. Net loss: {self.ticket_cost - ev:.2f} RSD per ticket "
                f"({-roi:.1f}% house edge)."
            )
        }

    def kelly_criterion(self, bankroll):
        """
        Kelly Criterion for optimal bet sizing.

        For negative EV games, Kelly says bet ZERO.
        But we provide a responsible entertainment budget.
        """
        ev = EXPECTED_VALUE_PER_TICKET
        edge = (ev - self.ticket_cost) / self.ticket_cost

        kelly_fraction = 0.0  # Always 0 for negative EV
        kelly_bet = 0.0

        # Entertainment budget: never more than 1-2% of disposable income
        entertainment_max_pct = 0.02
        entertainment_budget = bankroll * entertainment_max_pct

        max_tickets = int(entertainment_budget / self.ticket_cost)

        return {
            'kelly_fraction': kelly_fraction,
            'kelly_bet': kelly_bet,
            'kelly_says': 'DO NOT BET (negative expected value)',
            'edge': edge,
            'entertainment_budget': entertainment_budget,
            'entertainment_budget_pct': entertainment_max_pct * 100,
            'max_tickets_recommended': max_tickets,
            'bankroll': bankroll,
            'expected_loss': max_tickets * (self.ticket_cost - ev),
            'interpretation': (
                f"Kelly Criterion: bet 0 RSD (negative EV). "
                f"Entertainment budget (2% of {bankroll:,.0f}): {entertainment_budget:,.0f} RSD = "
                f"{max_tickets} tickets. Expected loss: {max_tickets * (self.ticket_cost - ev):,.0f} RSD."
            )
        }

    def simulate_long_term(self, n_tickets_per_draw, n_draws, n_simulations=10000):
        """
        Monte Carlo simulation of long-term outcomes.
        Shows the realistic distribution of wins and losses.
        """
        import numpy as np
        rng = np.random.default_rng(42)

        total_costs = n_tickets_per_draw * n_draws * self.ticket_cost
        final_balances = []

        all_numbers = list(range(1, MAX_NUMBER + 1))

        for sim in range(n_simulations):
            total_won = 0

            for draw in range(n_draws):
                # Simulate actual draw
                drawn = set(rng.choice(all_numbers, size=NUMBERS_PER_DRAW, replace=False))

                for ticket in range(n_tickets_per_draw):
                    # Generate random ticket
                    ticket_nums = set(rng.choice(
                        all_numbers, size=NUMBERS_PER_DRAW, replace=False
                    ))
                    matches = len(ticket_nums & drawn)
                    total_won += self.prize_table.get(matches, 0)

            net = total_won - total_costs
            final_balances.append(net)

        final_balances = np.array(final_balances)

        return {
            'n_simulations': n_simulations,
            'n_tickets_per_draw': n_tickets_per_draw,
            'n_draws': n_draws,
            'total_cost': total_costs,
            'mean_net': float(np.mean(final_balances)),
            'median_net': float(np.median(final_balances)),
            'std_net': float(np.std(final_balances)),
            'best_outcome': float(np.max(final_balances)),
            'worst_outcome': float(np.min(final_balances)),
            'pct_profitable': float(np.mean(final_balances > 0) * 100),
            'pct_break_even': float(np.mean(final_balances >= -total_costs * 0.1) * 100),
            'percentile_5': float(np.percentile(final_balances, 5)),
            'percentile_25': float(np.percentile(final_balances, 25)),
            'percentile_75': float(np.percentile(final_balances, 75)),
            'percentile_95': float(np.percentile(final_balances, 95)),
            'interpretation': (
                f"Over {n_draws} draws with {n_tickets_per_draw} tickets each "
                f"(cost: {total_costs:,.0f} RSD): "
                f"Average net: {np.mean(final_balances):,.0f} RSD. "
                f"Only {np.mean(final_balances > 0) * 100:.1f}% of simulations were profitable."
            )
        }

    def responsible_play_advice(self, monthly_income, monthly_play_budget=None):
        """Generate responsible gambling advice"""
        if monthly_play_budget is None:
            monthly_play_budget = monthly_income * 0.02  # 2% max

        draws_per_month = 12  # ~3 per week * 4 weeks
        budget_per_draw = monthly_play_budget / draws_per_month
        tickets_per_draw = int(budget_per_draw / self.ticket_cost)

        monthly_expected_loss = (
            tickets_per_draw * draws_per_month *
            (self.ticket_cost - EXPECTED_VALUE_PER_TICKET)
        )

        return {
            'monthly_income': monthly_income,
            'recommended_monthly_budget': monthly_play_budget,
            'budget_pct_of_income': monthly_play_budget / monthly_income * 100,
            'budget_per_draw': budget_per_draw,
            'tickets_per_draw': tickets_per_draw,
            'monthly_expected_loss': monthly_expected_loss,
            'annual_expected_loss': monthly_expected_loss * 12,
            'rules': [
                f"Nikada ne trošite više od {monthly_play_budget:,.0f} RSD mesečno na loto.",
                f"To je {tickets_per_draw} tiket(a) po izvlačenju.",
                "Tretirajte ovo kao zabavu, ne kao investiciju.",
                "Nikada ne pozajmljujte novac za igranje.",
                "Ako gubite više nego što možete priuštiti, prestanite.",
                f"Očekivani mesečni gubitak: {monthly_expected_loss:,.0f} RSD.",
                "Nijedan sistem ne može da savlada matematiku lutrije.",
            ]
        }