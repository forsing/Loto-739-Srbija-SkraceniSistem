"""
Loto Serbia scraper - v3.3
Scrapes ONLY Loto 7/39 results (gameNo=1)
Filters out other games (Loto Plus, Džoker, instant lotteries)
"""
import requests
import re
import io
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from lotto_ai.config import logger, MAX_NUMBER, MIN_NUMBER, NUMBERS_PER_DRAW, IS_CLOUD
from lotto_ai.core.db import get_session, Draw

RESULTS_URLS = [
    "https://www.lutrija.rs/Results?gameNo=1",
    "https://lutrija.rs/Results?gameNo=1",
]

OLD_BASE_URL = "https://lutrija.rs/Results/OfficialReports?gameNo=1"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]


def _get_session():
    """Create requests session with retry"""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "sr-RS,sr;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })

    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def _fetch_page(url, timeout=30):
    """Fetch page with retry"""
    session = _get_session()

    for t in [timeout, timeout + 15]:
        try:
            logger.debug(f"Fetching {url} (timeout={t}s)")
            response = session.get(url, timeout=t)
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout):
            logger.warning(f"Timeout {url} ({t}s), retrying...")
            continue
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error: {e}")
            continue
        except Exception as e:
            logger.warning(f"Error: {e}")
            break

    logger.error(f"All attempts failed for {url}")
    return None


# ============================================================================
# HTML SCRAPER - LOTO 7/39 ONLY
# ============================================================================

def scrape_results_page():
    """
    Scrape the LATEST Loto 7/39 draw from lutrija.rs/Results?gameNo=1
    
    The page shows multiple games. We take ONLY the FIRST valid
    Loto 7/39 result (the latest draw shown at the top).
    
    Returns:
        list with at most 1 draw dict, or empty list
    """
    response = None
    for url in RESULTS_URLS:
        logger.info(f"Trying: {url}")
        response = _fetch_page(url)
        if response:
            break

    if not response:
        logger.error("Could not reach lutrija.rs")
        return []

    try:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find ALL title labels on the page
        title_labels = soup.select('div.Rez_Txt_Title label')

        if not title_labels:
            for label in soup.find_all('label'):
                text = label.get_text(strip=True)
                if 'коло' in text and ('извлачења' in text or 'датум' in text):
                    title_labels.append(label)

        logger.info(f"Found {len(title_labels)} total titles on page")

        # Find the FIRST title that matches Loto 7/39 format
        for title_label in title_labels:
            text = title_label.get_text(strip=True)

            # SKIP instant lotteries
            if re.search(r'извлачење\s+\d{4}', text):
                logger.debug(f"Skip instant: {text[:60]}")
                continue
            if 'време извлачења' in text:
                logger.debug(f"Skip timed: {text[:60]}")
                continue

            # Must match standard format
            match = re.match(
                r'Извештај\s+за\s+(\d+)\.\s*коло\s*-\s*датум\s+извлачења\s+(\d{2})\.(\d{2})\.(\d{4})',
                text
            )
            if not match:
                logger.debug(f"Skip non-matching: {text[:60]}")
                continue

            round_number = int(match.group(1))
            day, month, year = match.group(2), match.group(3), match.group(4)
            draw_date = f"{year}-{month}-{day}"

            try:
                datetime.strptime(draw_date, '%Y-%m-%d')
            except ValueError:
                continue

            # This is our Loto 7/39 title — find its numbers
            # The numbers are in the SAME parent container as this title
            # We need to find the SPECIFIC number block for THIS game section
            
            numbers = _find_numbers_for_first_game(title_label)

            if numbers and len(numbers) == NUMBERS_PER_DRAW and validate_numbers(numbers):
                logger.info(f"✅ Loto 7/39: {draw_date} (kolo {round_number}): {numbers}")
                
                # Return ONLY this one draw — it's the latest Loto 7/39
                return [{
                    'round_number': round_number,
                    'draw_date': draw_date,
                    'numbers': numbers
                }]
            else:
                logger.warning(f"Found title but bad numbers for {draw_date}: {numbers}")
                # Don't continue to next title — the next one is a different game
                # Instead, try harder to find numbers for THIS title
                numbers_fallback = _find_numbers_fallback(soup)
                if numbers_fallback:
                    logger.info(f"✅ Loto 7/39 (fallback): {draw_date} "
                              f"(kolo {round_number}): {numbers_fallback}")
                    return [{
                        'round_number': round_number,
                        'draw_date': draw_date,
                        'numbers': numbers_fallback
                    }]

        logger.warning("No valid Loto 7/39 draw found on page")
        return []

    except Exception as e:
        logger.error(f"Error parsing results: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def _find_numbers_for_first_game(title_element):
    """
    Find the 7 numbers for the FIRST game section on the page.
    
    The page structure is:
        div (game section)
            div.Rez_Txt_Title > label "Извештај за 16. коло..."
            ...
            div.float_left.width_100
                div.float_left > div.Rez_Brojevi_Txt_Gray  (number 1)
                div.float_left > div.Rez_Brojevi_Txt_Gray  (number 2)
                ...
    
    We find the CLOSEST div.float_left.width_100 that contains
    exactly 7 number divs AFTER this title.
    """
    # Method 1: Find next sibling/descendant containers with numbers
    current = title_element
    
    # Walk through next elements after the title
    for elem in title_element.find_all_next():
        # If we hit another title, STOP — we've left this game's section
        if elem.name and elem.get('class'):
            classes = ' '.join(elem.get('class', []))
            if 'Rez_Txt_Title' in classes:
                break
        
        # Look for the container with numbers
        if (elem.name == 'div' and elem.get('class') and 
            'float_left' in elem.get('class', []) and 
            'width_100' in elem.get('class', [])):
            
            number_divs = elem.select('div.Rez_Brojevi_Txt_Gray')
            if number_divs:
                numbers = []
                for div in number_divs:
                    text = div.get_text(strip=True)
                    try:
                        num = int(text)
                        if MIN_NUMBER <= num <= MAX_NUMBER:
                            numbers.append(num)
                    except (ValueError, TypeError):
                        continue
                
                if len(numbers) == NUMBERS_PER_DRAW:
                    return sorted(numbers)
    
    # Method 2: Walk up to parent, find first number container
    parent = title_element.parent
    for _ in range(8):
        if parent is None:
            break
        
        # Find ALL width_100 containers in this parent
        containers = parent.select('div.float_left.width_100')
        for container in containers:
            number_divs = container.select('div.Rez_Brojevi_Txt_Gray')
            if number_divs:
                numbers = []
                for div in number_divs:
                    text = div.get_text(strip=True)
                    try:
                        num = int(text)
                        if MIN_NUMBER <= num <= MAX_NUMBER:
                            numbers.append(num)
                    except (ValueError, TypeError):
                        continue
                
                if len(numbers) == NUMBERS_PER_DRAW:
                    return sorted(numbers)
        
        parent = parent.parent
    
    return []


def _find_numbers_fallback(soup):
    """
    Last resort: find the FIRST set of exactly 7 numbers (1-39) on the page.
    This works because Loto 7/39 is the first game shown.
    """
    # Find ALL number divs on the page
    all_number_divs = soup.select('div.Rez_Brojevi_Txt_Gray')
    
    if not all_number_divs:
        return None
    
    # Group consecutive number divs
    numbers = []
    for div in all_number_divs:
        text = div.get_text(strip=True)
        try:
            num = int(text)
            if MIN_NUMBER <= num <= MAX_NUMBER:
                numbers.append(num)
            else:
                # Number out of range — might be start of different game
                if len(numbers) == NUMBERS_PER_DRAW:
                    return sorted(numbers)
                numbers = []
        except (ValueError, TypeError):
            if len(numbers) == NUMBERS_PER_DRAW:
                return sorted(numbers)
            numbers = []
        
        if len(numbers) == NUMBERS_PER_DRAW:
            return sorted(numbers)
    
    if len(numbers) == NUMBERS_PER_DRAW:
        return sorted(numbers)
    
    return None


def _find_loto739_numbers(title_element):
    """
    Find exactly 7 numbers (1-39) associated with a Loto 7/39 title.
    Walks up the DOM to find the containing section, then extracts
    numbers from Rez_Brojevi_Txt_Gray divs.
    """
    # Walk up to find the section containing both title and numbers
    parent = title_element.parent
    for _ in range(15):
        if parent is None:
            break

        number_divs = parent.select('div.Rez_Brojevi_Txt_Gray')
        if number_divs:
            numbers = []
            for div in number_divs:
                text = div.get_text(strip=True)
                try:
                    num = int(text)
                    if MIN_NUMBER <= num <= MAX_NUMBER:
                        numbers.append(num)
                except (ValueError, TypeError):
                    continue

            # Loto 7/39 has exactly 7 numbers
            if len(numbers) == NUMBERS_PER_DRAW:
                return sorted(numbers)

            # If we found numbers but wrong count, this might be wrong section
            # Keep walking up
            if len(numbers) > 0 and len(numbers) != NUMBERS_PER_DRAW:
                # Could be we're in a section with multiple games
                # Try to find a sub-section with exactly 7
                sub_sections = parent.select('div.float_left.width_100')
                for sub in sub_sections:
                    sub_divs = sub.select('div.Rez_Brojevi_Txt_Gray')
                    if sub_divs:
                        sub_nums = []
                        for div in sub_divs:
                            text = div.get_text(strip=True)
                            try:
                                num = int(text)
                                if MIN_NUMBER <= num <= MAX_NUMBER:
                                    sub_nums.append(num)
                            except (ValueError, TypeError):
                                continue

                        if len(sub_nums) == NUMBERS_PER_DRAW:
                            return sorted(sub_nums)

        parent = parent.parent

    # Strategy 2: find_all_next from title, stop at next title
    numbers = []
    for elem in title_element.find_all_next():
        # Stop at next title section
        if hasattr(elem, 'get') and elem.get('class'):
            classes = ' '.join(elem.get('class', []))
            if 'Rez_Txt_Title' in classes and elem != title_element.parent:
                break

        if hasattr(elem, 'get') and elem.get('class'):
            classes = ' '.join(elem.get('class', []))
            if 'Rez_Brojevi_Txt_Gray' in classes:
                text = elem.get_text(strip=True)
                try:
                    num = int(text)
                    if MIN_NUMBER <= num <= MAX_NUMBER:
                        numbers.append(num)
                except (ValueError, TypeError):
                    continue

        if len(numbers) == NUMBERS_PER_DRAW:
            return sorted(numbers)

        # Safety: don't go too far
        if len(numbers) > NUMBERS_PER_DRAW:
            return sorted(numbers[:NUMBERS_PER_DRAW])

    return sorted(numbers) if len(numbers) == NUMBERS_PER_DRAW else []


# ============================================================================
# MAIN SCRAPE FUNCTION
# ============================================================================

def scrape_recent_draws(max_pdfs=50):
    """
    Scrape recent Loto 7/39 draws.
    Returns number of new draws inserted.
    """
    inserted_count = 0

    # Step 1: HTML scraper
    logger.info("Trying HTML results scraper...")
    html_results = scrape_results_page()

    if html_results:
        session = get_session()
        try:
            for result in html_results:
                draw_date = result['draw_date']
                numbers = result['numbers']
                round_number = result.get('round_number')

                if not validate_numbers(numbers):
                    continue

                existing = session.query(Draw).filter_by(draw_date=draw_date).first()
                if existing:
                    if round_number and not existing.round_number:
                        existing.round_number = round_number
                        session.commit()
                        logger.info(f"Updated round_number for {draw_date}")
                    else:
                        logger.debug(f"Draw {draw_date} already exists, skipping")
                    continue

                draw = Draw(
                    draw_date=draw_date,
                    round_number=round_number,
                    n1=numbers[0], n2=numbers[1], n3=numbers[2],
                    n4=numbers[3], n5=numbers[4], n6=numbers[5],
                    n7=numbers[6]
                )
                session.add(draw)
                session.commit()
                inserted_count += 1
                logger.info(f"✅ Inserted: {draw_date} (kolo {round_number}): {numbers}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving: {e}")
        finally:
            session.close()

    if inserted_count > 0:
        logger.info(f"HTML scraper: {inserted_count} new draws")
        return inserted_count

    # Step 2: PDF fallback (local only)
    if not IS_CLOUD:
        logger.info("Trying PDF fallback...")
        return _scrape_from_pdfs(max_pdfs)
    else:
        logger.warning("No new draws found from cloud. Use manual input or update locally.")
        return 0


# ============================================================================
# MANUAL INPUT
# ============================================================================

def add_draw_manually(draw_date, numbers, round_number=None):
    """
    Manually add a draw to the database.
    Returns True if inserted, False otherwise.
    """
    numbers = sorted(numbers)

    if not validate_numbers(numbers):
        logger.error(f"Invalid numbers: {numbers}")
        return False

    try:
        datetime.strptime(draw_date, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date: {draw_date}")
        return False

    session = get_session()
    try:
        existing = session.query(Draw).filter_by(draw_date=draw_date).first()
        if existing:
            logger.info(f"Draw {draw_date} already exists")
            return False

        draw = Draw(
            draw_date=draw_date,
            round_number=round_number,
            n1=numbers[0], n2=numbers[1], n3=numbers[2],
            n4=numbers[3], n5=numbers[4], n6=numbers[5],
            n7=numbers[6]
        )
        session.add(draw)
        session.commit()
        logger.info(f"✅ Manual: {draw_date} (kolo {round_number}): {numbers}")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error: {e}")
        return False
    finally:
        session.close()


# ============================================================================
# FIX BAD DATA
# ============================================================================

def remove_bad_draws(dates_to_remove):
    """Remove incorrectly scraped draws"""
    session = get_session()
    removed = 0
    try:
        for date in dates_to_remove:
            draw = session.query(Draw).filter_by(draw_date=date).first()
            if draw:
                logger.info(f"Removing bad draw: {date} {draw.get_numbers()}")
                session.delete(draw)
                removed += 1
        session.commit()
        logger.info(f"Removed {removed} bad draws")
    except Exception as e:
        session.rollback()
        logger.error(f"Error removing draws: {e}")
    finally:
        session.close()
    return removed


# ============================================================================
# OLD PDF SCRAPER (for historical data)
# ============================================================================

def _scrape_from_pdfs(max_pdfs=50):
    """PDF-based scraper fallback"""
    js_data = extract_js_data()
    if not js_data:
        return 0

    session = get_session()
    inserted_count = 0

    try:
        for report in js_data[:max_pdfs]:
            pdf_path = report.get('OfficialReportPath')
            if not pdf_path:
                continue

            result = extract_numbers_from_pdf(pdf_path)
            if not result:
                continue

            round_number, draw_date, numbers = result

            existing = session.query(Draw).filter_by(draw_date=draw_date).first()
            if existing:
                if round_number and not existing.round_number:
                    existing.round_number = round_number
                    session.commit()
                continue

            draw = Draw(
                draw_date=draw_date, round_number=round_number,
                n1=numbers[0], n2=numbers[1], n3=numbers[2],
                n4=numbers[3], n5=numbers[4], n6=numbers[5],
                n7=numbers[6]
            )
            session.add(draw)
            session.commit()
            inserted_count += 1
    except Exception as e:
        session.rollback()
        logger.error(f"PDF error: {e}")
    finally:
        session.close()

    return inserted_count


def extract_js_data():
    response = _fetch_page(OLD_BASE_URL, timeout=15)
    if not response:
        return []
    try:
        match = re.search(r'var officialReportsTableData = (\[.*?\]);',
                          response.text, re.DOTALL)
        if not match:
            return []
        return json.loads(match.group(1))
    except Exception as e:
        logger.error(f"JS data error: {e}")
        return []


def extract_numbers_from_pdf(pdf_url):
    if PdfReader is None:
        return None

    response = _fetch_page(f"https://lutrija.rs{pdf_url}", timeout=20)
    if not response:
        return None

    try:
        reader = PdfReader(io.BytesIO(response.content))
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"

        round_number = None
        for pattern in [r'(\d+)[\s.]*(?:редовно\s+)?(?:коло|kolo)']:
            m = re.search(pattern, pdf_url, re.IGNORECASE) or \
                re.search(pattern, text, re.IGNORECASE)
            if m:
                round_number = int(m.group(1))
                break
        if not round_number:
            return None

        draw_date = None
        m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', pdf_url)
        if m:
            d, mo, y = m.groups()
            draw_date = f"{y}-{mo}-{d}"
        if not draw_date:
            m = re.search(r'од\s+(\d{2})\.(\d{2})\.(\d{4})', text[:500])
            if m:
                d, mo, y = m.groups()
                draw_date = f"{y}-{mo}-{d}"
        if not draw_date:
            return None

        sections = text.split('ЏОКЕР')
        early = sections[0][:800] if len(sections) > 1 else text[:800]
        all_nums = re.findall(r'\b([1-9]|[12]\d|3[0-9])\b', early)

        seen = []
        for s in all_nums:
            n = int(s)
            if MIN_NUMBER <= n <= MAX_NUMBER and n not in seen:
                seen.append(n)
            if len(seen) == 7:
                break

        if len(seen) == 7 and validate_numbers(seen):
            return round_number, draw_date, sorted(seen)

        return None
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return None


def validate_numbers(numbers):
    if len(numbers) != NUMBERS_PER_DRAW:
        return False
    if any(n < MIN_NUMBER or n > MAX_NUMBER for n in numbers):
        return False
    if len(set(numbers)) != NUMBERS_PER_DRAW:
        return False
    return True


parse_pdf_for_numbers = extract_numbers_from_pdf
scrape_all_draws = lambda: _scrape_from_pdfs(max_pdfs=1500)


if __name__ == "__main__":
    from lotto_ai.core.db import init_db
    init_db()
    print("Testing Loto 7/39 scraper...")
    n = scrape_recent_draws(max_pdfs=5)
    print(f"Inserted {n} new draws")