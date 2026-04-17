import os
import sys
from pathlib import Path

# -- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent


def _dataset_from_argv(argv: list[str]) -> str | None:
    """Extract `--dataset <name>` from argv when present."""
    for i, token in enumerate(argv):
        if token == "--dataset" and i + 1 < len(argv):
            value = argv[i + 1].strip()
            if value:
                return value
        if token.startswith("--dataset="):
            value = token.split("=", 1)[1].strip()
            if value:
                return value
    return None


def _resolve_data_dir() -> Path:
    """
    Resolve active dataset directory with priority:
      1) NEXTPOI_DATA_DIR (explicit full path)
      2) --dataset <name> CLI argument
      3) NEXTPOI_DATASET env var
      4) default datasets/nyc
    """
    if os.environ.get("NEXTPOI_DATA_DIR"):
        return Path(os.environ["NEXTPOI_DATA_DIR"])

    ds = _dataset_from_argv(sys.argv) or os.environ.get("NEXTPOI_DATASET")
    if ds:
        return BASE_DIR / "datasets" / ds

    return BASE_DIR / "datasets" / "nyc"


DATA_DIR = _resolve_data_dir()


def _dataset_tag(path: Path) -> str:
    """
    Build a stable cache/results namespace from dataset directory name.
    Examples:
      datasets/nyc -> nyc
      datasets/Gowalla-CA -> gowalla_ca
    """
    tag = path.resolve().name.lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in tag).strip("_")
    return safe or "default"


DATASET_TAG = _dataset_tag(DATA_DIR)
CACHE_DIR = BASE_DIR / "cache" / DATASET_TAG
RESULTS_DIR = BASE_DIR / "results" / DATASET_TAG

TRIPS_TRAIN = DATA_DIR / "trips_train.csv"
TRIPS_VALID = DATA_DIR / "trips_valid.csv"
TRIPS_TEST = DATA_DIR / "trips_test.csv"
LOC2ID_PATH = DATA_DIR / "loc2id"
USER_INDEX_PATH = DATA_DIR / "user_index.json"
PROMPTS_REFINED_DIR = DATA_DIR / "prompts_refined"

PROFILES_CACHE_DIR = CACHE_DIR / "profiles"
SIMILARITY_CACHE = CACHE_DIR / "similarity.pkl"
PREDICTIONS_CACHE_DIR = CACHE_DIR / "predictions"
TRANSITIONS_CACHE = CACHE_DIR / "transitions.pkl"
EVALUATION_RESULTS = RESULTS_DIR / "evaluation_results.json"

# -- LLM models ------------------------------------------------------------
PROFILE_LLM_MODEL = "gpt-5.4"
PREDICTION_LLM_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

API_MAX_RETRIES = 3
API_RETRY_BACKOFF = 5

# -- Profile Building ------------------------------------------------------
PROFILE_MAX_TRIPS = 10
PROFILE_MAX_CHECKINS = 8

# -- Similarity ------------------------------------------------------------
GEOHASH_PRECISION = 5
SPATIAL_WEIGHT = 0.5
EMBEDDING_WEIGHT = 0.5
TOP_K_SIMILAR_USERS = 5

# -- Temporal Pattern Mining -----------------------------------------------
HOUR_BUCKET_SIZE = 3
TOP_TRANSITIONS = 20

# -- Candidate Selection ---------------------------------------------------
SPATIAL_TOP_N = 100
FORCED_INCLUDE_N = 3
MAX_CANDIDATES = 100

DIST_WEIGHT = 0.4
CATEGORY_WEIGHT = 0.3
COLLAB_WEIGHT = 0.2
REVISIT_WEIGHT = 0.1

# -- Pre-filtering (LLM-based intent filter) -------------------------------
PREFILTER_TOP_N = 30
INTENT_LLM_MODEL = "gpt-5.4"
PREFILTER_LLM_MODEL = "gpt-5.4"
RAW_POOL_SIMILAR_TOP_N = 50
INTENT_CACHE_DIR = CACHE_DIR / "intent"
PREFILTER_CACHE_DIR = CACHE_DIR / "prefilter"

# -- Prediction ------------------------------------------------------------
MAX_CONTEXT_CHECKINS = 8

# -- Evaluation ------------------------------------------------------------
EVAL_K_VALUES = [1, 5, 10]
