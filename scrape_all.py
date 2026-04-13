"""
Full historical scrape with progress tracking
"""
import sys
from pathlib import Path
import time
from datetime import datetime

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from lotto_ai.scraper.serbia_scraper import extract_js_data, extract_numbers_from_pdf
from lotto_ai.core.db import get_session, Draw, init_db
from lotto_ai.config import logger, NUMBER_RANGE


def scrape_all_with_progress():
    """Scrape all available draws with detailed progress tracking"""
    print("=" * 70)
    print("🇷🇸 LOTO SERBIA - FULL HISTORICAL SCRAPE")
    print("=" * 70)

    init_db()

    print("\n📡 Fetching PDF list from lutrija.rs...")
    js_data = extract_js_data()

    if not js_data:
        print("❌ Failed to fetch PDF list")
        return

    total_pdfs = len(js_data)
    print(f"✅ Found {total_pdfs} official reports")

    stats = {
        'total': total_pdfs,
        'processed': 0,
        'inserted': 0,
        'already_exists': 0,
        'failed': 0,
        'start_time': time.time()
    }

    session = get_session()

    print(f"\n{'=' * 70}")
    print(f"📥 STARTING SCRAPE")
    print(f"{'=' * 70}")
    print(f"⏱️  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    min_num, max_num = NUMBER_RANGE

    try:
        for i, report in enumerate(js_data, 1):
            pdf_path = report.get('OfficialReportPath')
            if not pdf_path:
                stats['failed'] += 1
                continue

            if i % 10 == 0 or i == 1:
                elapsed = time.time() - stats['start_time']
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total_pdfs - i) / rate if rate > 0 else 0
                print(
                    f"📊 Progress: {i}/{total_pdfs} ({i / total_pdfs * 100:.1f}%) | "
                    f"✅ {stats['inserted']} inserted | "
                    f"⏭️  {stats['already_exists']} skipped | "
                    f"❌ {stats['failed']} failed | "
                    f"⏱️  ETA: {eta / 60:.1f} min"
                )

            result = extract_numbers_from_pdf(pdf_path)
            stats['processed'] += 1

            if not result:
                stats['failed'] += 1
                continue

            round_number, draw_date, numbers = result

            # Validate numbers
            if not all(min_num <= n <= max_num for n in numbers):
                logger.warning(f"Invalid numbers for {draw_date}: {numbers}")
                stats['failed'] += 1
                continue

            if len(set(numbers)) != 7:
                logger.warning(f"Duplicate numbers for {draw_date}: {numbers}")
                stats['failed'] += 1
                continue

            existing = session.query(Draw).filter_by(draw_date=draw_date).first()
            if existing:
                if existing.round_number is None and round_number is not None:
                    existing.round_number = round_number
                    session.commit()
                stats['already_exists'] += 1
                continue

            try:
                draw = Draw(
                    draw_date=draw_date,
                    round_number=round_number,
                    n1=numbers[0],
                    n2=numbers[1],
                    n3=numbers[2],
                    n4=numbers[3],
                    n5=numbers[4],
                    n6=numbers[5],
                    n7=numbers[6]
                )
                session.add(draw)
                session.commit()
                stats['inserted'] += 1

                if stats['inserted'] % 50 == 0:
                    logger.info(f"🎯 Milestone: {stats['inserted']} draws inserted!")

            except Exception as e:
                session.rollback()
                logger.error(f"Error inserting draw {draw_date}: {e}")
                stats['failed'] += 1

    except KeyboardInterrupt:
        print("\n\n⚠️  INTERRUPTED BY USER")
        session.commit()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        session.rollback()
    finally:
        session.close()

    elapsed = time.time() - stats['start_time']
    print(f"\n{'=' * 70}")
    print(f"📊 SCRAPING COMPLETE")
    print(f"{'=' * 70}")
    print(f"⏱️  Duration: {elapsed / 60:.1f} minutes")
    print(f"📥 Processed: {stats['processed']}/{stats['total']} PDFs")
    print(f"✅ Inserted: {stats['inserted']} new draws")
    print(f"⏭️  Skipped: {stats['already_exists']} (already in database)")
    print(f"❌ Failed: {stats['failed']}")
    print(f"{'=' * 70}")

    session = get_session()
    total_in_db = session.query(Draw).count()
    session.close()
    print(f"\n💾 Total draws in database: {total_in_db}")

    return stats


if __name__ == "__main__":
    scrape_all_with_progress()