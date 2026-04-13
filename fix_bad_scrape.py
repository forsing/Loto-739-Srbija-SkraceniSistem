"""
Remove the 3 incorrectly scraped draws from other games.
Run once, then delete this file.
"""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from lotto_ai.core.db import init_db, get_session, Draw
from lotto_ai.scraper.serbia_scraper import remove_bad_draws

init_db()

# These were scraped from other games (not Loto 7/39):
bad_dates = [
    '2026-02-25',  # Loto Plus (kolo 8)
    '2026-02-26',  # Different game (kolo 16 same as 24th = wrong)
    '2026-02-27',  # Instant lottery (kolo 2, multiple extractions)
]

print("Current latest draws:")
session = get_session()
draws = session.query(Draw).order_by(Draw.draw_date.desc()).limit(10).all()
for d in draws:
    kolo = f" (kolo {d.round_number})" if d.round_number else ""
    print(f"  {d.draw_date}{kolo}: {d.get_numbers()}")
session.close()

print(f"\nRemoving {len(bad_dates)} bad draws: {bad_dates}")
removed = remove_bad_draws(bad_dates)
print(f"Removed: {removed}")

print("\nAfter fix:")
session = get_session()
draws = session.query(Draw).order_by(Draw.draw_date.desc()).limit(10).all()
for d in draws:
    kolo = f" (kolo {d.round_number})" if d.round_number else ""
    print(f"  {d.draw_date}{kolo}: {d.get_numbers()}")
total = session.query(Draw).count()
session.close()
print(f"\nTotal draws: {total}")