"""
Rigorous statistical fairness tests for lottery draws.
Tests whether the lottery shows any exploitable deviation from uniformity.
"""
import numpy as np
import pandas as pd
from math import comb
from datetime import datetime
from scipy import stats as scipy_stats
from collections import Counter
import json
import itertools
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lotto_ai.config import (
    logger, NUMBER_RANGE, NUMBERS_PER_DRAW, MAX_NUMBER, MIN_NUMBER,
    TOTAL_COMBINATIONS, MATCH_PROBABILITIES
)
from lotto_ai.core.db import get_session, FairnessTest, Draw


class LotteryFairnessAnalyzer:
    """
    Comprehensive statistical analysis of lottery fairness.
    If the lottery is fair (it almost certainly is), no prediction
    method can beat random selection.
    """

    def __init__(self):
        self.results = {}

    def load_draws(self):
        """Load all draws as list of number lists"""
        session = get_session()
        try:
            draws = session.query(Draw).order_by(Draw.draw_date).all()
            return [d.get_numbers() for d in draws], [d.draw_date for d in draws]
        finally:
            session.close()

    def run_all_tests(self, save_to_db=True):
        """Run complete fairness test suite"""
        draws, dates = self.load_draws()

        if len(draws) < 30:
            logger.warning(f"Only {len(draws)} draws - need 30+ for reliable tests")
            return {"error": "Insufficient data", "n_draws": len(draws)}

        results = {}
        results['n_draws'] = len(draws)
        results['date_range'] = f"{dates[0]} to {dates[-1]}"

        # Test 1: Chi-square uniformity test
        results['chi_square'] = self._chi_square_uniformity(draws)

        # Test 2: Pairs frequency test
        results['pairs_test'] = self._pairs_frequency_test(draws)

        # Test 3: Serial correlation test
        results['serial_correlation'] = self._serial_correlation_test(draws)

        # Test 4: Runs test for randomness
        results['runs_test'] = self._runs_test(draws)

        # Test 5: Gap analysis
        results['gap_analysis'] = self._gap_distribution_test(draws)

        # Test 6: Sum distribution test
        results['sum_test'] = self._sum_distribution_test(draws)

        # Test 7: Odd/even distribution test
        results['odd_even_test'] = self._odd_even_test(draws)

        # Test 8: Consecutive numbers test
        results['consecutive_test'] = self._consecutive_numbers_test(draws)

        # Overall verdict
        p_values = []
        for key, val in results.items():
            if isinstance(val, dict) and 'p_value' in val:
                if val['p_value'] is not None:
                    p_values.append(val['p_value'])

        if p_values:
            # Bonferroni correction for multiple testing
            n_tests = len(p_values)
            adjusted_alpha = 0.05 / n_tests
            any_significant = any(p < adjusted_alpha for p in p_values)

            results['overall'] = {
                'n_tests': n_tests,
                'bonferroni_alpha': adjusted_alpha,
                'min_p_value': min(p_values),
                'any_significant_after_correction': any_significant,
                'verdict': 'SUSPICIOUS - possible pattern detected' if any_significant
                           else 'FAIR - no exploitable patterns detected',
                'recommendation': (
                    'Statistical anomaly detected - investigate further'
                    if any_significant
                    else 'Lottery appears fair. Coverage optimization is your best strategy.'
                )
            }
        else:
            results['overall'] = {
                'verdict': 'INSUFFICIENT DATA',
                'recommendation': 'Need more draws for reliable analysis'
            }

        if save_to_db:
            self._save_results(results)

        self.results = results
        return results

    def _chi_square_uniformity(self, draws):
        """
        Chi-square goodness-of-fit test.
        H0: All numbers equally likely.
        """
        all_numbers = [n for draw in draws for n in draw]
        n_total = len(all_numbers)

        observed = np.zeros(MAX_NUMBER)
        for n in all_numbers:
            observed[n - 1] += 1

        expected_freq = n_total / MAX_NUMBER
        expected = np.full(MAX_NUMBER, expected_freq)

        chi2_stat, p_value = scipy_stats.chisquare(observed, expected)

        return {
            'test_name': 'Chi-Square Uniformity',
            'statistic': float(chi2_stat),
            'p_value': float(p_value),
            'degrees_of_freedom': MAX_NUMBER - 1,
            'conclusion': 'FAIR' if p_value > 0.05 else 'DEVIATION DETECTED',
            'expected_frequency': float(expected_freq),
            'most_frequent': int(np.argmax(observed) + 1),
            'most_frequent_count': int(np.max(observed)),
            'least_frequent': int(np.argmin(observed) + 1),
            'least_frequent_count': int(np.min(observed)),
            'interpretation': (
                f"p={p_value:.4f}. "
                f"{'No significant deviation from uniform distribution.' if p_value > 0.05 else 'Significant deviation detected - but could be due to sample size.'}"
            )
        }

    def _pairs_frequency_test(self, draws):
        """Test if any pair of numbers appears together more than expected"""
        pair_counts = Counter()
        for draw in draws:
            for pair in itertools.combinations(sorted(draw), 2):
                pair_counts[pair] += 1

        n_draws = len(draws)
        # Expected pair frequency: C(5,5)*C(37,5) ... simplified:
        # P(both i,j drawn) = C(37,5)/C(39,7) * ... = 7/39 * 6/38
        expected_pair_freq = n_draws * (NUMBERS_PER_DRAW / MAX_NUMBER) * (
            (NUMBERS_PER_DRAW - 1) / (MAX_NUMBER - 1)
        )

        pair_freqs = list(pair_counts.values())

        if len(pair_freqs) < 10:
            return {
                'test_name': 'Pairs Frequency',
                'p_value': None,
                'conclusion': 'INSUFFICIENT DATA'
            }

        # Chi-square on pair frequencies
        total_possible_pairs = comb(MAX_NUMBER, 2)
        observed_pairs = np.zeros(total_possible_pairs)
        all_pairs = list(itertools.combinations(range(1, MAX_NUMBER + 1), 2))

        for idx, pair in enumerate(all_pairs):
            observed_pairs[idx] = pair_counts.get(pair, 0)

        expected_arr = np.full(total_possible_pairs, expected_pair_freq)

        # Only test if expected > 5 (chi-square requirement)
        if expected_pair_freq >= 1:
            chi2_stat, p_value = scipy_stats.chisquare(observed_pairs, expected_arr)
        else:
            chi2_stat = 0
            p_value = 1.0

        most_common_pair = pair_counts.most_common(1)[0] if pair_counts else ((0, 0), 0)

        return {
            'test_name': 'Pairs Frequency',
            'statistic': float(chi2_stat),
            'p_value': float(p_value),
            'expected_pair_frequency': float(expected_pair_freq),
            'most_common_pair': list(most_common_pair[0]),
            'most_common_pair_count': int(most_common_pair[1]),
            'conclusion': 'FAIR' if p_value > 0.05 else 'ANOMALY DETECTED',
            'interpretation': f"Most common pair {most_common_pair[0]} appeared {most_common_pair[1]} times (expected ~{expected_pair_freq:.1f})"
        }

    def _serial_correlation_test(self, draws):
        """Test for serial correlation between consecutive draws"""
        if len(draws) < 10:
            return {'test_name': 'Serial Correlation', 'p_value': None, 'conclusion': 'INSUFFICIENT DATA'}

        # For each number, create binary time series and test autocorrelation
        correlations = []
        for number in range(MIN_NUMBER, MAX_NUMBER + 1):
            series = [1 if number in draw else 0 for draw in draws]
            if len(set(series)) < 2:
                continue
            # Lag-1 autocorrelation
            s = np.array(series, dtype=float)
            mean = s.mean()
            if np.std(s) == 0:
                continue
            autocorr = np.corrcoef(s[:-1], s[1:])[0, 1]
            if not np.isnan(autocorr):
                correlations.append(autocorr)

        if not correlations:
            return {'test_name': 'Serial Correlation', 'p_value': None, 'conclusion': 'INSUFFICIENT DATA'}

        mean_corr = np.mean(correlations)
        # Under H0 (independence), autocorrelations ~ N(0, 1/sqrt(n))
        se = 1.0 / np.sqrt(len(draws))
        z_stat = mean_corr / se
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

        return {
            'test_name': 'Serial Correlation',
            'statistic': float(z_stat),
            'p_value': float(p_value),
            'mean_autocorrelation': float(mean_corr),
            'standard_error': float(se),
            'conclusion': 'FAIR' if p_value > 0.05 else 'CORRELATION DETECTED',
            'interpretation': (
                f"Mean lag-1 autocorrelation: {mean_corr:.4f}. "
                f"{'No serial dependence detected.' if p_value > 0.05 else 'Possible serial dependence!'}"
            )
        }

    def _runs_test(self, draws):
        """Wald-Wolfowitz runs test for each number"""
        if len(draws) < 20:
            return {'test_name': 'Runs Test', 'p_value': None, 'conclusion': 'INSUFFICIENT DATA'}

        p_values = []
        for number in range(MIN_NUMBER, MAX_NUMBER + 1):
            series = [1 if number in draw else 0 for draw in draws]
            n1 = sum(series)
            n0 = len(series) - n1

            if n1 < 2 or n0 < 2:
                continue

            # Count runs
            runs = 1
            for i in range(1, len(series)):
                if series[i] != series[i - 1]:
                    runs += 1

            # Expected runs and variance
            n = n0 + n1
            expected_runs = (2 * n0 * n1) / n + 1
            var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))

            if var_runs <= 0:
                continue

            z = (runs - expected_runs) / np.sqrt(var_runs)
            p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
            p_values.append(p)

        if not p_values:
            return {'test_name': 'Runs Test', 'p_value': None, 'conclusion': 'INSUFFICIENT DATA'}

        # Fisher's method to combine p-values
        chi2_combined = -2 * sum(np.log(max(p, 1e-15)) for p in p_values)
        df = 2 * len(p_values)
        combined_p = 1 - scipy_stats.chi2.cdf(chi2_combined, df)

        return {
            'test_name': 'Runs Test (Combined)',
            'statistic': float(chi2_combined),
            'p_value': float(combined_p),
            'n_individual_tests': len(p_values),
            'n_individually_significant': sum(1 for p in p_values if p < 0.05),
            'conclusion': 'FAIR' if combined_p > 0.05 else 'NON-RANDOM PATTERN',
            'interpretation': (
                f"Combined runs test across all {len(p_values)} numbers. "
                f"{'Sequence appears random.' if combined_p > 0.05 else 'Non-random sequence detected!'}"
            )
        }

    def _gap_distribution_test(self, draws):
        """Test if gaps between appearances follow geometric distribution"""
        all_gaps = []

        for number in range(MIN_NUMBER, MAX_NUMBER + 1):
            last_seen = None
            for i, draw in enumerate(draws):
                if number in draw:
                    if last_seen is not None:
                        all_gaps.append(i - last_seen)
                    last_seen = i

        if len(all_gaps) < 30:
            return {'test_name': 'Gap Distribution', 'p_value': None, 'conclusion': 'INSUFFICIENT DATA'}

        # Under H0, gaps ~ Geometric(p=7/39)
        p_success = NUMBERS_PER_DRAW / MAX_NUMBER
        expected_mean_gap = 1.0 / p_success
        observed_mean_gap = np.mean(all_gaps)

        # KS test against geometric distribution
        theoretical_cdf = lambda x: 1 - (1 - p_success) ** x
        ks_stat, ks_p = scipy_stats.kstest(all_gaps, theoretical_cdf)

        return {
            'test_name': 'Gap Distribution',
            'statistic': float(ks_stat),
            'p_value': float(ks_p),
            'expected_mean_gap': float(expected_mean_gap),
            'observed_mean_gap': float(observed_mean_gap),
            'n_gaps_analyzed': len(all_gaps),
            'conclusion': 'FAIR' if ks_p > 0.05 else 'ANOMALOUS GAPS',
            'interpretation': (
                f"Expected mean gap: {expected_mean_gap:.2f}, observed: {observed_mean_gap:.2f}. "
                f"{'Gap distribution consistent with fair lottery.' if ks_p > 0.05 else 'Unusual gap pattern detected.'}"
            )
        }

    def _sum_distribution_test(self, draws):
        """Test if the sum of drawn numbers follows expected distribution"""
        sums = [sum(draw) for draw in draws]

        # Expected sum of 7 numbers drawn without replacement from 1..39
        # E[sum] = 7 * (1+39)/2 = 7 * 20 = 140
        expected_mean = NUMBERS_PER_DRAW * (MIN_NUMBER + MAX_NUMBER) / 2

        # Var[sum] for hypergeometric-like sampling
        pop_var = np.var(range(MIN_NUMBER, MAX_NUMBER + 1))
        n = NUMBERS_PER_DRAW
        N = MAX_NUMBER
        expected_var = n * pop_var * (N - n) / (N - 1)
        expected_std = np.sqrt(expected_var)

        observed_mean = np.mean(sums)
        observed_std = np.std(sums)

        # Shapiro-Wilk for normality (CLT should make sums approximately normal)
        if len(sums) >= 20:
            # Use sample if too many draws
            test_sums = sums[:5000] if len(sums) > 5000 else sums
            sw_stat, sw_p = scipy_stats.shapiro(test_sums)
        else:
            sw_stat, sw_p = 0, 1.0

        # Z-test for mean
        z_stat = (observed_mean - expected_mean) / (expected_std / np.sqrt(len(sums)))
        z_p = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

        return {
            'test_name': 'Sum Distribution',
            'statistic': float(z_stat),
            'p_value': float(z_p),
            'expected_mean_sum': float(expected_mean),
            'observed_mean_sum': float(observed_mean),
            'expected_std': float(expected_std),
            'observed_std': float(observed_std),
            'normality_p': float(sw_p),
            'conclusion': 'FAIR' if z_p > 0.05 else 'BIASED SUMS',
            'interpretation': (
                f"Expected mean sum: {expected_mean:.1f}, observed: {observed_mean:.1f}. "
                f"{'Sum distribution is consistent with fair drawing.' if z_p > 0.05 else 'Sum distribution shows bias.'}"
            )
        }

    def _odd_even_test(self, draws):
        """Test if odd/even distribution matches expected"""
        n_odd_per_draw = [sum(1 for n in draw if n % 2 == 1) for draw in draws]

        # Under H0: hypergeometric distribution
        # 20 odd numbers (1,3,...,39) and 19 even numbers (2,4,...,38)
        n_odd_total = len([n for n in range(MIN_NUMBER, MAX_NUMBER + 1) if n % 2 == 1])
        n_even_total = MAX_NUMBER - n_odd_total

        # Expected: 7 * 20/39 = 3.59 odd numbers per draw
        expected_odd = NUMBERS_PER_DRAW * n_odd_total / MAX_NUMBER

        observed_mean = np.mean(n_odd_per_draw)

        # Chi-square on distribution of odd counts
        observed_dist = Counter(n_odd_per_draw)
        categories = list(range(0, NUMBERS_PER_DRAW + 1))

        observed_arr = np.array([observed_dist.get(k, 0) for k in categories], dtype=float)

        # Expected from hypergeometric
        expected_arr = np.array([
            scipy_stats.hypergeom.pmf(k, MAX_NUMBER, n_odd_total, NUMBERS_PER_DRAW) * len(draws)
            for k in categories
        ])

        # Merge bins with expected < 5
        obs_merged = []
        exp_merged = []
        obs_acc = 0
        exp_acc = 0
        for o, e in zip(observed_arr, expected_arr):
            obs_acc += o
            exp_acc += e
            if exp_acc >= 5:
                obs_merged.append(obs_acc)
                exp_merged.append(exp_acc)
                obs_acc = 0
                exp_acc = 0
        if exp_acc > 0:
            if exp_merged:
                obs_merged[-1] += obs_acc
                exp_merged[-1] += exp_acc
            else:
                obs_merged.append(obs_acc)
                exp_merged.append(exp_acc)

        if len(obs_merged) >= 2:
            chi2_stat, p_value = scipy_stats.chisquare(obs_merged, exp_merged)
        else:
            chi2_stat, p_value = 0, 1.0

        return {
            'test_name': 'Odd/Even Distribution',
            'statistic': float(chi2_stat),
            'p_value': float(p_value),
            'expected_odd_per_draw': float(expected_odd),
            'observed_mean_odd': float(observed_mean),
            'n_odd_in_pool': n_odd_total,
            'n_even_in_pool': n_even_total,
            'conclusion': 'FAIR' if p_value > 0.05 else 'ODD/EVEN BIAS',
            'interpretation': f"Expected {expected_odd:.2f} odd numbers per draw, observed {observed_mean:.2f}"
        }

    def _consecutive_numbers_test(self, draws):
        """Test if consecutive number pairs appear at expected rate"""
        n_consecutive_per_draw = []
        for draw in draws:
            s = sorted(draw)
            consec = sum(1 for i in range(len(s) - 1) if s[i + 1] == s[i] + 1)
            n_consecutive_per_draw.append(consec)

        # Monte Carlo estimation of expected consecutive pairs
        rng = np.random.default_rng(42)
        mc_consecutive = []
        for _ in range(10000):
            sample = sorted(rng.choice(range(MIN_NUMBER, MAX_NUMBER + 1),
                                       size=NUMBERS_PER_DRAW, replace=False))
            consec = sum(1 for i in range(len(sample) - 1) if sample[i + 1] == sample[i] + 1)
            mc_consecutive.append(consec)

        expected_mean = np.mean(mc_consecutive)
        observed_mean = np.mean(n_consecutive_per_draw)

        # Two-sample t-test
        t_stat, p_value = scipy_stats.ttest_ind(n_consecutive_per_draw, mc_consecutive)

        return {
            'test_name': 'Consecutive Numbers',
            'statistic': float(t_stat),
            'p_value': float(p_value),
            'expected_mean_consecutive': float(expected_mean),
            'observed_mean_consecutive': float(observed_mean),
            'conclusion': 'FAIR' if p_value > 0.05 else 'CONSECUTIVE ANOMALY',
            'interpretation': (
                f"Expected {expected_mean:.2f} consecutive pairs per draw, observed {observed_mean:.2f}. "
                f"{'Normal.' if p_value > 0.05 else 'Unusual consecutive number pattern!'}"
            )
        }

    def _save_results(self, results):
        """Save test results to database"""
        session = get_session()
        try:
            now = datetime.now().isoformat()
            n_draws = results.get('n_draws', 0)

            for key, val in results.items():
                if isinstance(val, dict) and 'test_name' in val:
                    ft = FairnessTest(
                        tested_at=now,
                        test_name=val['test_name'],
                        statistic_value=val.get('statistic'),
                        p_value=val.get('p_value'),
                        conclusion=val.get('conclusion', ''),
                        n_draws_tested=n_draws,
                        details=json.dumps(val)
                    )
                    session.add(ft)

            session.commit()
            logger.info(f"Saved {len([v for v in results.values() if isinstance(v, dict) and 'test_name' in v])} test results")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving test results: {e}")
        finally:
            session.close()

    def get_exploitability_score(self):
        """
        Return a score 0-100 indicating how exploitable the lottery appears.
        0 = perfectly fair (not exploitable)
        100 = highly anomalous (possibly exploitable)
        """
        if not self.results or 'overall' not in self.results:
            self.run_all_tests()

        p_values = []
        for key, val in self.results.items():
            if isinstance(val, dict) and 'p_value' in val:
                if val['p_value'] is not None:
                    p_values.append(val['p_value'])

        if not p_values:
            return 0

        # Score based on how many tests are borderline or significant
        score = 0
        for p in p_values:
            if p < 0.01:
                score += 100 / len(p_values)
            elif p < 0.05:
                score += 50 / len(p_values)
            elif p < 0.10:
                score += 20 / len(p_values)

        return min(100, score)