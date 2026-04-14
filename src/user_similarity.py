"""
Stage 2: Compute pairwise user similarity and find top-K similar users.

Similarity = 0.5 * geohash_jaccard + 0.5 * profile_embedding_cosine

Results are cached to SIMILARITY_CACHE (pickle).
"""
import pickle

import numpy as np

from src.config import (
    EMBEDDING_MODEL,
    GEOHASH_PRECISION,
    SIMILARITY_CACHE,
    SPATIAL_WEIGHT,
    EMBEDDING_WEIGHT,
    TOP_K_SIMILAR_USERS,
)
from src.utils import Progress, logger

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model {EMBEDDING_MODEL}…")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


# ── Geohash Jaccard ────────────────────────────────────────────────────────

def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _build_spatial_matrix(profiles: dict[str, dict]) -> tuple[list[str], np.ndarray]:
    """Return (user_ids, N×N geohash Jaccard matrix)."""
    user_ids = list(profiles.keys())
    N = len(user_ids)
    gh_sets = [set(profiles[uid].get("visited_geohashes", [])) for uid in user_ids]

    mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i, N):
            s = _jaccard(gh_sets[i], gh_sets[j])
            mat[i, j] = s
            mat[j, i] = s
    return user_ids, mat


# ── Profile embedding cosine ───────────────────────────────────────────────

def _get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts locally using sentence-transformers (BAAI/bge-m3).
    Returns L2-normalised float32 array of shape (N, EMBEDDING_DIM).
    """
    model = _get_embed_model()
    BATCH_SIZE = 32
    all_vecs = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vecs = model.encode(batch, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs)
        logger.info(f"Embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
    return np.vstack(all_vecs).astype(np.float32)


def _build_embedding_matrix(
    user_ids: list[str], profiles: dict[str, dict]
) -> np.ndarray:
    """Return N×N cosine similarity matrix by averaging enhanced_profile and raw_stats_prompt embeddings."""
    enhanced_texts = [profiles[uid].get("enhanced_profile", "") for uid in user_ids]
    stats_texts = [profiles[uid].get("raw_stats_prompt", "") for uid in user_ids]
    embs_enhanced = _get_embeddings(enhanced_texts)  # (N, EMBEDDING_DIM) normalised
    embs_stats = _get_embeddings(stats_texts)        # (N, EMBEDDING_DIM) normalised
    # Average the two embeddings then re-normalise
    embs_avg = embs_enhanced + embs_stats
    norms = np.linalg.norm(embs_avg, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embs_avg = embs_avg / norms
    mat = embs_avg @ embs_avg.T  # cosine similarity, shape (N, N)
    np.clip(mat, -1.0, 1.0, out=mat)
    return mat.astype(np.float32)


# ── Combined similarity ────────────────────────────────────────────────────

def build_similarity_index(
    profiles: dict[str, dict], force: bool = False
) -> dict[str, list[tuple[str, float]]]:
    """
    Compute pairwise user similarity and cache results.

    Returns: {user_id: [(similar_uid, score), …] sorted desc by score, top-K}
    """
    SIMILARITY_CACHE.parent.mkdir(parents=True, exist_ok=True)

    if not force and SIMILARITY_CACHE.exists():
        logger.info("Loading similarity index from cache…")
        with open(SIMILARITY_CACHE, "rb") as f:
            return pickle.load(f)

    logger.info(f"Building similarity matrix for {len(profiles)} users…")

    # Spatial
    logger.info("Computing geohash Jaccard matrix…")
    user_ids, spatial_mat = _build_spatial_matrix(profiles)

    # Embedding
    logger.info(f"Computing profile embedding matrix ({EMBEDDING_MODEL})…")
    embed_mat = _build_embedding_matrix(user_ids, profiles)

    # Combine
    combined = SPATIAL_WEIGHT * spatial_mat + EMBEDDING_WEIGHT * embed_mat
    np.fill_diagonal(combined, -1.0)  # exclude self

    # Build top-K lookup dict
    N = len(user_ids)
    top_k = min(TOP_K_SIMILAR_USERS, N - 1)
    index: dict[str, list[tuple[str, float]]] = {}
    for i, uid in enumerate(user_ids):
        row = combined[i]
        top_idxs = np.argsort(row)[::-1][:top_k]
        index[uid] = [(user_ids[j], float(row[j])) for j in top_idxs]

    with open(SIMILARITY_CACHE, "wb") as f:
        pickle.dump(index, f)
    logger.info(f"Similarity index saved to {SIMILARITY_CACHE}")
    return index


def get_similar_users(
    user_id: str,
    similarity_index: dict[str, list[tuple[str, float]]],
    k: int = TOP_K_SIMILAR_USERS,
) -> list[tuple[str, float]]:
    """Return top-k similar users as [(user_id, score), …]."""
    return similarity_index.get(user_id, [])[:k]
