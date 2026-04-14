"""Build and query prior memory from trajectory prefixes."""
from __future__ import annotations

import pickle
from collections import Counter

import numpy as np

from src.config import MAX_CONTEXT_CHECKINS, POI_EMBED_CACHE, PRIOR_BANK_CACHE
from src.embedding_utils import embed_texts
from src.utils import logger, movement_direction


def build_prefix_text(context: list[dict], data_loader, max_checkins: int = MAX_CONTEXT_CHECKINS) -> str:
    """Compact summary text for a prefix trajectory."""
    if not context:
        return "Empty trajectory prefix."

    if len(context) > max_checkins:
        head = context[:2]
        tail = context[-(max_checkins - 2):]
        display = head + [None] + tail
    else:
        display = context

    lines: list[str] = []
    for idx, item in enumerate(display, 1):
        if item is None:
            lines.append("... earlier stops omitted ...")
            continue
        name = data_loader.get_poi_name(item["loc_id"])
        cat = data_loader.get_poi_category(item["loc_id"])
        lines.append(f"{idx}. {item['date']} {item['time']} | {name} | {cat}")

    last = context[-1]
    lines.append(f"Last location category: {data_loader.get_poi_category(last['loc_id'])}")
    lines.append(f"Movement trend: {movement_direction(context)}")
    return "\n".join(lines)


def _build_poi_temporal_stats(data_loader) -> dict[int, dict]:
    temporal: dict[int, dict] = {}
    for tid in data_loader.train_traj_ids + data_loader.valid_traj_ids:
        for c in data_loader.trips[tid]:
            lid = c["loc_id"]
            slot = temporal.setdefault(lid, {"total": 0, "by_dow": {}, "by_hb": {}})
            slot["total"] += 1
            dow = str(c["weekday"])
            hb = str(c["hour"] // 3)
            slot["by_dow"][dow] = slot["by_dow"].get(dow, 0) + 1
            slot["by_hb"][hb] = slot["by_hb"].get(hb, 0) + 1
    return temporal


def build_or_load_poi_embeddings(data_loader, force: bool = False) -> dict:
    """Cache POI text embeddings and popularity/temporal statistics."""
    POI_EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not force and POI_EMBED_CACHE.exists():
        with open(POI_EMBED_CACHE, "rb") as f:
            payload = pickle.load(f)
        if "temporal_stats" not in payload:
            payload["temporal_stats"] = _build_poi_temporal_stats(data_loader)
            with open(POI_EMBED_CACHE, "wb") as f:
                pickle.dump(payload, f)
        logger.info(f"Loaded POI embeddings from cache ({len(payload['loc_ids'])} POIs).")
        return payload

    logger.info("Building POI embedding cache...")
    loc_ids = list(data_loader.all_loc_ids)
    texts = [data_loader.get_poi_description(lid) for lid in loc_ids]
    vectors = embed_texts(texts)
    categories = [data_loader.get_poi_category(lid) for lid in loc_ids]

    popularity = Counter()
    for tid in data_loader.train_traj_ids + data_loader.valid_traj_ids:
        for c in data_loader.trips[tid]:
            popularity[c["loc_id"]] += 1
    pop_vec = np.array([popularity.get(lid, 0) for lid in loc_ids], dtype=np.float32)

    payload = {
        "loc_ids": loc_ids,
        "vectors": vectors,
        "categories": categories,
        "popularity": pop_vec,
        "temporal_stats": _build_poi_temporal_stats(data_loader),
        "loc_id_to_index": {lid: i for i, lid in enumerate(loc_ids)},
    }
    with open(POI_EMBED_CACHE, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"POI embeddings saved to {POI_EMBED_CACHE}")
    return payload


def build_prior_bank(data_loader, force: bool = False) -> dict:
    """Build an index of train/validation trajectory prefixes for retrieval."""
    PRIOR_BANK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not force and PRIOR_BANK_CACHE.exists():
        with open(PRIOR_BANK_CACHE, "rb") as f:
            payload = pickle.load(f)
        logger.info(f"Loaded prior bank from cache ({len(payload['meta'])} examples).")
        return payload

    traj_ids = list(data_loader.train_traj_ids) + list(data_loader.valid_traj_ids)
    meta: list[dict] = []
    texts: list[str] = []
    for tid in traj_ids:
        traj = data_loader.trips.get(tid, [])
        if len(traj) < 2:
            continue
        prefix = traj[:-1]
        target = traj[-1]
        user_id = next((uid for uid, tids in data_loader.trips_by_user.items() if tid in tids), None)
        texts.append(build_prefix_text(prefix, data_loader))
        meta.append({
            "traj_id": tid,
            "user_id": user_id,
            "target_loc_id": target["loc_id"],
            "target_category": data_loader.get_poi_category(target["loc_id"]),
            "last_loc_id": prefix[-1]["loc_id"],
            "last_category": data_loader.get_poi_category(prefix[-1]["loc_id"]),
            "prefix_len": len(prefix),
        })

    vectors = embed_texts(texts)
    payload = {
        "vectors": vectors.astype(np.float32),
        "meta": meta,
    }
    with open(PRIOR_BANK_CACHE, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Prior bank saved to {PRIOR_BANK_CACHE} ({len(meta)} examples).")
    return payload


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        return vec / max(np.linalg.norm(vec), 1e-12)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vec / norms


class PriorBankIndex:
    def __init__(self, payload: dict):
        self.vectors = _normalize(payload["vectors"])
        self.meta = list(payload["meta"])
        self.traj_to_index = {m["traj_id"]: i for i, m in enumerate(self.meta)}

    def retrieve(self, prefix_vector: np.ndarray, top_k: int = 12, exclude_traj_id: str | None = None):
        vec = _normalize(prefix_vector)
        sims = self.vectors @ vec
        if exclude_traj_id and exclude_traj_id in self.traj_to_index:
            sims[self.traj_to_index[exclude_traj_id]] = -1e9
        top_k = min(top_k, len(sims))
        idx = np.argpartition(sims, -top_k)[-top_k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        top_sims = sims[idx]

        support_by_loc: dict[int, dict[str, float]] = {}
        for bank_idx, sim in zip(idx, top_sims):
            loc_id = int(self.meta[bank_idx]["target_loc_id"])
            bucket = support_by_loc.setdefault(loc_id, {"count": 0.0, "max_sim": -1e9, "sum_sim": 0.0})
            bucket["count"] += 1.0
            bucket["sum_sim"] += float(sim)
            bucket["max_sim"] = max(bucket["max_sim"], float(sim))
        for bucket in support_by_loc.values():
            bucket["mean_sim"] = bucket["sum_sim"] / max(bucket["count"], 1.0)
        return {
            "indices": idx,
            "similarities": top_sims,
            "support_by_loc": support_by_loc,
        }


def load_prior_bank_index(payload: dict | None = None) -> PriorBankIndex | None:
    if payload is None:
        if not PRIOR_BANK_CACHE.exists():
            return None
        with open(PRIOR_BANK_CACHE, "rb") as f:
            payload = pickle.load(f)
    return PriorBankIndex(payload)
