"""
Verify scraped data quality - Enhanced with statistical checks
"""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from lotto_ai.core.db import get_session, Draw
from lotto_ai.config import NUMBER_RANGE, NUMBERS_PER_DRAW
from collections import Counter
import numpy as np

session = get_session()
draws = session.query(Draw).order_by(Draw.draw_date.desc()).all()
session.close()

min_num, max_num = NUMBER_RANGE

print("=" * 70)
print("📊 DATA QUALITY REPORT")
print("=" * 70)

print(f"\n📈 Total Draws: {len(draws)}")

if draws:
    dates = [d.draw_date for d in draws]
    print(f"📅 Date Range: {min(dates)} to {max(dates)}")

    # Check for draws with round numbers
    with_round = sum(1 for d in draws if d.round_number is not None)
    print(f"🔢 Draws with round number: {with_round}/{len(draws)}")

    # Sample draws
    print(f"\n🎲 Latest 10 Draws:")
    print("-" * 70)
    for draw in draws[:10]:
        numbers = draw.get_numbers()
        kolo = f" (Kolo {draw.round_number})" if draw.round_number else ""
        print(f"  {draw.draw_date}{kolo}: {numbers}")

    # Number validation
    all_numbers = []
    issues = []
    for draw in draws:
        nums = draw.get_numbers()
        all_numbers.extend(nums)

        # Check range
        for n in nums:
            if n < min_num or n > max_num:
                issues.append(f"  ❌ {draw.draw_date}: number {n} out of range")

        # Check uniqueness
        if len(set(nums)) != NUMBERS_PER_DRAW:
            issues.append(f"  ❌ {draw.draw_date}: duplicate numbers {nums}")

        # Check count
        if len(nums) != NUMBERS_PER_DRAW:
            issues.append(f"  ❌ {draw.draw_date}: wrong count {len(nums)}")

    print(f"\n📊 Number Statistics:")
    print(f"   Total numbers drawn: {len(all_numbers)}")
    print(f"   Unique numbers used: {len(set(all_numbers))}")
    print(f"   Min number: {min(all_numbers)}")
    print(f"   Max number: {max(all_numbers)}")

    if issues:
        print(f"\n⚠️  DATA ISSUES ({len(issues)}):")
        for issue in issues[:20]:
            print(issue)
    else:
        print(f"   ✅ All numbers valid (range {min_num}-{max_num}, {NUMBERS_PER_DRAW} per draw)")

    # Frequency analysis
    freq = Counter(all_numbers)
    expected_freq = len(all_numbers) / (max_num - min_num + 1)

    print(f"\n🔥 Top 10 Most Frequent:")
    for num, count in freq.most_common(10):
        deviation = (count - expected_freq) / expected_freq * 100
        print(f"   {num:2d}: {count:3d} times ({deviation:+.1f}% vs expected)")

    print(f"\n❄️  Top 10 Least Frequent:")
    for num, count in freq.most_common()[-10:]:
        deviation = (count - expected_freq) / expected_freq * 100
        print(f"   {num:2d}: {count:3d} times ({deviation:+.1f}% vs expected)")

    # Chi-square quick check
    observed = np.array([freq.get(n, 0) for n in range(min_num, max_num + 1)])
    expected = np.full(max_num - min_num + 1, expected_freq)
    chi2 = np.sum((observed - expected) ** 2 / expected)
    df = max_num - min_num
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df)

    print(f"\n📐 Quick Chi-Square Test:")
    print(f"   Chi²: {chi2:.2f} (df={df})")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Verdict: {'✅ FAIR (uniform)' if p_value > 0.05 else '⚠️ POSSIBLE DEVIATION'}")

    # Duplicate date check
    date_counts = Counter(dates)
    duplicates = {d: c for d, c in date_counts.items() if c > 1}
    if duplicates:
        print(f"\n⚠️  Duplicate dates found:")
        for date, count in duplicates.items():
            print(f"   {date}: {count} entries")
    else:
        print(f"\n✅ No duplicate dates")

else:
    print("❌ No draws found in database!")

print("=" * 70)