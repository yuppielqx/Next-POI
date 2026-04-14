"""Shared text embedding utilities."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.config import EMBEDDING_MODEL
from src.utils import logger


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model {EMBEDDING_MODEL}...")
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed texts with BAAI/bge-m3 and return L2-normalized float32 vectors."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model = get_embedding_model()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors.append(vecs)
        logger.info(f"Embedded {min(start + batch_size, len(texts))}/{len(texts)} texts")
    return np.vstack(vectors).astype(np.float32)
