from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "foursquare_NYC"
CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"

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
PRIOR_BANK_CACHE = CACHE_DIR / "prior_bank.pkl"
POI_EMBED_CACHE = CACHE_DIR / "poi_embeddings.pkl"
RERANKER_MODEL_PATH = CACHE_DIR / "latent_reranker.pt"
RERANKER_META_PATH = CACHE_DIR / "latent_reranker_meta.json"
EVALUATION_RESULTS = RESULTS_DIR / "evaluation_results.json"

# ── LLM models ─────────────────────────────────────────────────────────────
PROFILE_LLM_MODEL = "gpt-5.4"
PREDICTION_LLM_MODEL = "gpt-5.4-mini"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

API_MAX_RETRIES = 3
API_RETRY_BACKOFF = 5

# ── Profile Building ───────────────────────────────────────────────────────
PROFILE_MAX_TRIPS = 10
PROFILE_MAX_CHECKINS = 8

# ── Similarity ─────────────────────────────────────────────────────────────
GEOHASH_PRECISION = 5
SPATIAL_WEIGHT = 0.5
EMBEDDING_WEIGHT = 0.5
TOP_K_SIMILAR_USERS = 5

# ── Temporal Pattern Mining ────────────────────────────────────────────────
HOUR_BUCKET_SIZE = 3
TOP_TRANSITIONS = 20

# ── Candidate Selection ────────────────────────────────────────────────────
SPATIAL_TOP_N = 100
FORCED_INCLUDE_N = 3
MAX_CANDIDATES = 100

DIST_WEIGHT = 0.4
CATEGORY_WEIGHT = 0.3
COLLAB_WEIGHT = 0.2
REVISIT_WEIGHT = 0.1

# ── Pre-filtering (LLM-based intent filter) ────────────────────────────────
PREFILTER_TOP_N = 30
INTENT_LLM_MODEL = "gpt-5.4"
PREFILTER_LLM_MODEL = "gpt-5.4"
RAW_POOL_SIMILAR_TOP_N = 50
INTENT_CACHE_DIR = CACHE_DIR / "intent"
PREFILTER_CACHE_DIR = CACHE_DIR / "prefilter"

# ── Learned reranker ───────────────────────────────────────────────────────
PRIOR_BANK_TOP_K = 12
RERANKER_HIDDEN_DIM = 256
RERANKER_NEGATIVES = 24
RERANKER_BATCH_SIZE = 32
RERANKER_EPOCHS = 4
RERANKER_LR = 2e-4
RERANKER_MAX_PREFIX_CHECKINS = 8

# ── Prediction ─────────────────────────────────────────────────────────────
MAX_CONTEXT_CHECKINS = 8
HYBRID_TOP_N = 15

# ── Evaluation ─────────────────────────────────────────────────────────────
EVAL_K_VALUES = [1, 5, 10]
