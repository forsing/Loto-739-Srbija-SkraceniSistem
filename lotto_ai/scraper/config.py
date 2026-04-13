"""
Configuration for Lotto AI
Production-ready with environment variables
"""
from pathlib import Path
import os
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment detection
IS_CLOUD = os.getenv("CLOUD_ENV", "0") == "1"
IS_RAILWAY = os.getenv("RAILWAY_ENVIRONMENT") is not None
IS_STREAMLIT = os.getenv("STREAMLIT_RUNTIME") is not None

# Base directories
if IS_RAILWAY:
    BASE_DIR = Path("/app/data")
elif IS_STREAMLIT or IS_CLOUD:
    BASE_DIR = Path("/tmp")
else:
    # Local development
    BASE_DIR = Path(__file__).resolve().parent.parent / "data"

# Ensure data directory exists
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Database path
DB_PATH = BASE_DIR / "lotto_max.db"

# Models directory
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Scraping configuration
SCRAPING_ENABLED = not IS_STREAMLIT  # Disable scraping on Streamlit Cloud
BASE_URL = "https://loteries.lotoquebec.com/en/lotteries/lotto-max-resultats"

logger.info(f"Environment: Cloud={IS_CLOUD}, Railway={IS_RAILWAY}, Streamlit={IS_STREAMLIT}")
logger.info(f"Data directory: {BASE_DIR}")
logger.info(f"Database path: {DB_PATH}")
logger.info(f"Scraping enabled: {SCRAPING_ENABLED}")