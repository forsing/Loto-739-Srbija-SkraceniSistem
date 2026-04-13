"""
Configuration for Loto Serbia Portfolio Optimizer - v4.0
"""
import os
import logging
from pathlib import Path
from math import comb

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================
IS_RAILWAY = os.getenv("RAILWAY_ENVIRONMENT") is not None
IS_STREAMLIT_CLOUD = (
    os.getenv("STREAMLIT_SHARING_MODE") is not None or
    os.getenv("STREAMLIT_RUNTIME_ENVIRONMENT") == "cloud" or
    os.path.exists("/mount/src")
)
IS_CLOUD = IS_RAILWAY or IS_STREAMLIT_CLOUD

# ============================================================================
# PATHS
# ============================================================================
if IS_STREAMLIT_CLOUD:
    BASE_DIR = Path("/mount/src/loto-serbia-ai")
    DATA_DIR = BASE_DIR / "data"
elif IS_RAILWAY:
    BASE_DIR = Path("/app")
    DATA_DIR = BASE_DIR / "data"
else:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "loto_serbia.db"
CSV_DRAWS_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4596_k29.csv")

# ============================================================================
# LOTTERY CONFIGURATION - SERBIA LOTO 7/39
# ============================================================================
MIN_NUMBER = 1
MAX_NUMBER = 39
NUMBERS_PER_DRAW = 7
NUMBER_RANGE = (MIN_NUMBER, MAX_NUMBER)
VALID_NUMBERS = list(range(MIN_NUMBER, MAX_NUMBER + 1))

DRAW_DAYS = [1, 4]  # Monday, Thursday - adjust if needed
DRAW_HOUR = 20
DRAW_MINUTE = 0
DRAW_TIMEZONE = "Europe/Belgrade"

GAME_NAME = "Loto 7/39"
GAME_COUNTRY = "Serbia"
GAME_ID = 1
DRAWS_PER_WEEK = 2

# ============================================================================
# PRIZE TABLE (RSD)
# NOTE: This is a simplified fixed table. If official payouts vary by pool,
# then EV is only approximate unless modeled by historical prize pools.
# ============================================================================
PRIZE_TABLE = {
    7: 10_000_000,
    6: 100_000,
    5: 1_500,
    4: 50,
    3: 20
}
TICKET_COST = 100

# ============================================================================
# MATHEMATICAL CONSTANTS
# ============================================================================
TOTAL_COMBINATIONS = comb(MAX_NUMBER, NUMBERS_PER_DRAW)

def _calc_ev():
    ev = 0.0
    for k, prize in PRIZE_TABLE.items():
        remaining = MAX_NUMBER - NUMBERS_PER_DRAW
        needed = NUMBERS_PER_DRAW - k
        if needed < 0 or needed > remaining:
            continue
        p = (comb(NUMBERS_PER_DRAW, k) * comb(remaining, needed)) / TOTAL_COMBINATIONS
        ev += p * prize
    return ev

EXPECTED_VALUE_PER_TICKET = _calc_ev()

# ============================================================================
# SCRAPING
# ============================================================================
BASE_URL = "https://lutrija.rs/Results/OfficialReports?gameNo=1"
SCRAPING_ENABLED = False
SCRAPE_INTERVAL_HOURS = 24
MAX_RETRIES = 3
TIMEOUT_SECONDS = 20

# ============================================================================
# PORTFOLIO / OPTIMIZER DEFAULTS
# ============================================================================
DEFAULT_TICKETS = 10
MAX_TICKETS = 50
MIN_DRAWS_FOR_ANALYSIS = 50
LOOKBACK_WINDOW = 20
COVERAGE_MONTE_CARLO_SAMPLES = 1500
WHEEL_GUARANTEE_DEFAULT = 3

# Score weights for coverage optimizer
DEFAULT_OPTIMIZER_WEIGHTS = {
    "w_pairs": 1.0,
    "w_triples": 0.30,
    "w_overlap": 3.0,
    "w_odd_even_penalty": 5.0,
    "w_sum_penalty": 3.0,
}

DEFAULT_OPTIMIZER_CONSTRAINTS = {
    "odd_min": 2,
    "odd_max": 5,
    "sum_tolerance_ratio": 0.30,
    "overlap_penalty_threshold": 5,
}

# ============================================================================
# MONTE CARLO / EVALUATION
# ============================================================================
PORTFOLIO_SIMULATIONS_DEFAULT = 20000
PORTFOLIO_SIMULATIONS_FAST = 5000
RANDOM_BASELINE_PORTFOLIOS = 500
BACKTEST_RANDOM_BASELINE_PORTFOLIOS = 1000
BACKTEST_BOOTSTRAP_SAMPLES = 5000
RNG_SEED = 42

# ============================================================================
# APP BRANDING
# ============================================================================
APP_TITLE = "Loto Serbia Portfolio Optimizer"
APP_SUBTITLE = "Coverage Optimization, Wheeling, and Honest Mathematics"
APP_VERSION = "4.0"

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = logging.DEBUG if not IS_CLOUD else logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("lotto_ai.config")

logger.info(f"Environment: Cloud={IS_CLOUD}, Railway={IS_RAILWAY}, Streamlit={IS_STREAMLIT_CLOUD}")
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Database path: {DB_PATH}")
logger.info(f"CSV draws path: {CSV_DRAWS_PATH}")
logger.info(f"Number range: {NUMBER_RANGE}")
logger.info(f"Total combinations: {TOTAL_COMBINATIONS:,}")
logger.info(f"Expected value per ticket: {EXPECTED_VALUE_PER_TICKET:.2f} RSD")
logger.info(f"Draw days: {DRAW_DAYS}")