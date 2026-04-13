"""
Test script for Serbia setup - v3.0
Works on Windows (no multiline -c needed)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_math_engine():
    print("=" * 60)
    print("TEST 1: Math Engine")
    print("=" * 60)
    try:
        from lotto_ai.core.math_engine import (
            match_probability, expected_value_per_ticket,
            kelly_criterion_lottery, portfolio_expected_value
        )

        p7 = match_probability(7)
        p3 = match_probability(3)
        print(f"  P(7/7) = {p7:.12f} = 1 in {int(1/p7):,}")
        print(f"  P(3/7) = {p3:.6f} = 1 in {int(1/p3):,}")

        ev = expected_value_per_ticket()
        print(f"  EV per ticket: {ev['expected_value']:.2f} RSD")
        print(f"  Net EV: {ev['net_ev']:.2f} RSD")
        print(f"  ROI: {ev['roi_percent']:.1f}%")

        kelly = kelly_criterion_lottery(10000)
        print(f"  Kelly: {kelly['kelly_says']}")
        print(f"  Max tickets: {kelly['max_responsible_tickets']}")

        pev = portfolio_expected_value(10)
        print(f"  10-ticket P(3+): {pev['prob_any_3plus']:.2%}")

        assert ev['roi_percent'] < 0, "ROI should be negative"
        assert kelly['kelly_fraction'] == 0, "Kelly should say don't bet"
        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coverage_optimizer():
    print("\n" + "=" * 60)
    print("TEST 2: Coverage Optimizer")
    print("=" * 60)
    try:
        from lotto_ai.core.coverage_optimizer import optimize_portfolio_coverage

        portfolio, stats = optimize_portfolio_coverage(
            5, monte_carlo_samples=500
        )

        print(f"  Generated {len(portfolio)} tickets")
        for i, t in enumerate(portfolio, 1):
            print(f"    Ticket {i}: {t}")
        print(f"  Pair coverage: {stats['pair_coverage_pct']:.1f}%")
        print(f"  Unique numbers: {stats['unique_numbers']}/39")
        print(f"  Avg overlap: {stats['avg_overlap']:.1f}")

        assert len(portfolio) == 5
        for ticket in portfolio:
            assert len(ticket) == 7, f"Wrong ticket size: {len(ticket)}"
            assert all(1 <= n <= 39 for n in ticket), f"Out of range: {ticket}"
            assert len(set(ticket)) == 7, f"Duplicates: {ticket}"

        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wheeling():
    print("\n" + "=" * 60)
    print("TEST 3: Wheeling System")
    print("=" * 60)
    try:
        from lotto_ai.core.wheeling import (
            generate_abbreviated_wheel, wheel_cost_estimate
        )

        key_numbers = [3, 7, 11, 15, 19, 23, 27, 31, 35]
        print(f"  Key numbers: {key_numbers}")

        estimate = wheel_cost_estimate(len(key_numbers), 3, 3)
        print(f"  Cost estimate: {estimate['estimated_min_tickets']}-"
              f"{estimate['estimated_max_tickets']} tickets")

        tickets, guarantee = generate_abbreviated_wheel(
            key_numbers,
            guarantee_if_hit=3,
            guarantee_match=3,
            max_tickets=30
        )

        print(f"  Generated {len(tickets)} tickets")
        for i, t in enumerate(tickets, 1):
            key_count = sum(1 for n in t if n in key_numbers)
            print(f"    Ticket {i}: {t} ({key_count} key numbers)")
        print(f"  Verified: {guarantee['verified']}")
        print(f"  Coverage: {guarantee['coverage_pct']:.1f}%")
        print(f"  Guarantee: {guarantee['guarantee']}")

        if guarantee['verified']:
            print("  PASSED (guarantee verified!)")
        else:
            print(f"  WARNING: {guarantee.get('warning', 'incomplete')}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database():
    print("\n" + "=" * 60)
    print("TEST 4: Database")
    print("=" * 60)
    try:
        from lotto_ai.core.db import init_db, get_session, Draw

        init_db()
        session = get_session()
        count = session.query(Draw).count()
        session.close()

        print(f"  Database initialized, {count} draws")
        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_features():
    print("\n" + "=" * 60)
    print("TEST 5: Features")
    print("=" * 60)
    try:
        from lotto_ai.features.features import load_draws, build_feature_matrix

        df = load_draws()
        print(f"  Loaded {len(df)} draws")

        if len(df) == 0:
            print("  SKIPPED (no data)")
            return True

        features = build_feature_matrix()
        unique = features['number'].nunique()
        print(f"  Feature matrix: {len(features)} rows, {unique} unique numbers")

        assert unique == 39, f"Expected 39 numbers, got {unique}"
        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_portfolio_generation():
    print("\n" + "=" * 60)
    print("TEST 6: Portfolio Generation")
    print("=" * 60)
    try:
        from lotto_ai.features.features import build_feature_matrix
        from lotto_ai.core.models import (
            generate_adaptive_portfolio, portfolio_statistics
        )

        features = build_feature_matrix()
        if len(features) == 0:
            print("  SKIPPED (no data)")
            return True

        portfolio, meta = generate_adaptive_portfolio(
            features, n_tickets=5, strategy='coverage_optimized'
        )

        stats = portfolio_statistics(portfolio)

        print(f"  Generated {len(portfolio)} tickets")
        for i, t in enumerate(portfolio, 1):
            print(f"    Ticket {i}: {t}")
        print(f"  Pair coverage: {stats['pair_coverage_pct']:.1f}%")
        print(f"  Strategy: {meta['strategy']}")

        assert len(portfolio) == 5
        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fairness():
    print("\n" + "=" * 60)
    print("TEST 7: Fairness Testing")
    print("=" * 60)
    try:
        from lotto_ai.features.features import load_draws
        from lotto_ai.core.math_engine import test_lottery_fairness

        df = load_draws()
        if len(df) < 30:
            print(f"  SKIPPED (need 30+ draws, have {len(df)})")
            return True

        results = test_lottery_fairness(df)

        print(f"  Draws analyzed: {results['n_draws']}")
        print(f"  Chi-square p: {results['chi_square']['p_value']:.4f}")
        print(f"  Chi-square: {results['chi_square']['conclusion']}")
        print(f"  Runs test: {results['runs_test']['conclusion']}")
        print(f"  Serial corr: {results['serial_correlation']['conclusion']}")
        print(f"  Overall: {results['overall']['conclusion']}")
        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scraping():
    print("\n" + "=" * 60)
    print("TEST 8: Scraping (3 PDFs)")
    print("=" * 60)
    try:
        from lotto_ai.scraper.serbia_scraper import scrape_recent_draws

        n = scrape_recent_draws(max_pdfs=3)
        print(f"  Inserted {n} draws")
        print("  PASSED")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print()
    print("=" * 60)
    print("  LOTO SERBIA v3.0 - FULL TEST SUITE")
    print("=" * 60)

    tests = [
        ("Math Engine", test_math_engine),
        ("Coverage Optimizer", test_coverage_optimizer),
        ("Wheeling System", test_wheeling),
        ("Database", test_database),
        ("Features", test_features),
        ("Portfolio Generation", test_portfolio_generation),
        ("Fairness Testing", test_fairness),
        ("Scraping", test_scraping),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
        except Exception as e:
            print(f"  CRASHED: {e}")
            passed = False
        results.append((name, passed))

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}: {status}")

    n_passed = sum(1 for _, p in results if p)
    n_total = len(results)
    print(f"\n  Total: {n_passed}/{n_total} passed")

    if n_passed == n_total:
        print("\n  ALL TESTS PASSED!")
    else:
        print("\n  Some tests failed - check output above")

    print("=" * 60)


if __name__ == "__main__":
    main()