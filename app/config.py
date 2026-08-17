"""
Shared configuration for BookLens: paths, taxonomies, models, and logging.
"""

import logging
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

DATASETS_DIR = PROJECT_ROOT / "datasets"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Catalog and vector store artifacts
CATALOG_CSV = DATA_DIR / "catalog.csv"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
VECTOR_STORE_COLLECTION = "booklens_catalog"

# Trained classifier artifacts
CATEGORY_MODEL_PATH = MODELS_DIR / "category_nn.npz"
EMOTION_MODEL_PATH = MODELS_DIR / "emotion_nn.npz"
METRICS_PATH = MODELS_DIR / "metrics.json"
EMOTION_PROFILE_PATH = MODELS_DIR / "catalog_average_emotion_profile.npy"

# ---------------------------------------------------------------------------
# Taxonomies & Schema
# ---------------------------------------------------------------------------

CATEGORIES = ["Academia", "Fiction", "Nepali Literature", "Nonfiction"]
EMOTIONS = ["fear", "joy", "neutral", "sadness"]

CANONICAL_COLUMNS = [
    "isbn13",
    "title",
    "authors",
    "categories",
    "description",
    "published_year",
    "average_rating",
    "num_pages",
    "ratings_count",
    "thumbnail",
]

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDING_PASSAGE_PREFIX = "passage: "
EMBEDDING_QUERY_PREFIX = "query: "

# Offline teacher models used for dataset labeling in notebook 04
TEACHER_MODEL_NAME = "MoritzLaurer/deberta-v3-large-mnli-fever-anli-ling-wanli"
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Lightweight zero-shot fallback for live queries in the app
ONLINE_ZERO_SHOT_MODEL_NAME = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"

# Optional API keys
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Theme Colors
# ---------------------------------------------------------------------------

ACCENT_COLOR = "#F97316"     # Selected / highlighted item (orange)
LOCAL_COLOR = "#3B82F6"      # Local catalog item (blue)
LIVE_COLOR = "#10B981"       # Live web item (green)
QUERY_COLOR = "#EF4444"      # Query vector (red)
CORPUS_COLOR = "#9CA3AF"     # Background corpus (gray)

_LIGHT_TEXT_COLOR = "#111827"
_DARK_TEXT_COLOR = "#E5E7EB"


def dark_mode_active() -> bool:
    """Check if Streamlit is currently set to dark theme."""
    import streamlit as st

    return st.context.theme.get("type") == "dark"


def chart_text_color(dark_mode: bool) -> str:
    """Return appropriate chart text color based on active theme."""
    return _DARK_TEXT_COLOR if dark_mode else _LIGHT_TEXT_COLOR


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_THIRD_PARTY_LOGGERS = (
    "httpx",
    "huggingface_hub",
    "chromadb",
    "sentence_transformers",
)

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _quiet_third_party_loggers() -> None:
    for lib_name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(lib_name).setLevel(logging.WARNING)


def is_present(value) -> bool:
    """Check if a field value is non-null, non-empty, and not NaN."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def build_passage_text(title, categories, description) -> str:
    """Construct unified passage text for book embedding."""
    parts = [str(value).strip() for value in (title, categories, description) if is_present(value)]
    return " . ".join(parts)


def get_logger(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    """Return a configured logger writing to console and log file."""
    logger = logging.getLogger(f"booklens.{name}")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    _quiet_third_party_loggers()

    return logger
