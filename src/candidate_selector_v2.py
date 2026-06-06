"""
Stage 3a V2: Two-source quota candidate selection with prior-guided scoring.

Key differences from v1 (candidate_selector.py):
  - Two sources only: history + nearby (similar_user removed as candidate
    source — near-zero GT recall at specific POI level; kept as LLM context)
  - Each source scored by its own signal:
      history: freq + dow + hb + 0.5×trans
      nearby:  NEARBY_DIST_SCALE / dist_km
  - Intent prior's likely_categories / likely_specific_places add bonus scores
  - Fixed per-source quotas (24 history / 6 nearby) replace heuristic global sort
"""
from collections import Counter

from src.config import (
    HOUR_BUCKET_SIZE,
    MAX_CANDIDATES,
    NEARBY_DIST_SCALE,
    QUOTA_HISTORY,
    QUOTA_NEARBY,
    SPATIAL_TOP_N,
)
from src.utils import haversine

_HB_LABELS = [
    "midnight–2am", "3–5am", "6–8am", "9–11am",
    "noon–2pm", "3–5pm", "6–8pm", "9–11pm",
]
_DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def build_raw_pool(
    context: list[dict],
    user_profile: dict,
    _similar_users: list[tuple[str, float]],
    data_loader,
    ablation: str = "full",
) -> list[dict]:
    """
    Collect raw candidates from two sources: history and nearby.
    similar_user is excluded as a candidate source (near-zero GT recall
    at the specific POI level); it remains useful as LLM reasoning context.

    ablation="spatial_only" skips history; ablation="history_only" skips nearby.

    Returns list of dicts:
      {loc_id, name, category, dist_km, snippet, source, visit_count}
    """
    if not context:
        return []

    last = context[-1]
    query_lon, query_lat = last["lon"], last["lat"]

    visited_loc_ids: list[int] = user_profile.get("visited_loc_ids", [])
    visit_freq = Counter(visited_loc_ids)

    dist_cache: dict[int, float] = {}

    def get_dist(lid: int) -> float:
        if lid not in dist_cache:
            if lid in data_loader.id2loc:
                plon, plat = data_loader.id2loc[lid]
                dist_cache[lid] = haversine(query_lon, query_lat, plon, plat)
            else:
                dist_cache[lid] = 999.0
        return dist_cache[lid]

    pool: list[dict] = []
    seen: set[int] = set()

    # ── User's own history ────────────────────────────────────────────────
    if ablation != "spatial_only":
        for loc_id, count in visit_freq.items():
            meta = data_loader.get_poi_metadata(loc_id)
            pool.append({
                "loc_id": loc_id,
                "name": meta["name"],
                "category": meta["category"],
                "dist_km": get_dist(loc_id),
                "snippet": data_loader.get_poi_snippet(loc_id, sentences=2),
                "source": "history",
                "visit_count": count,
            })
            seen.add(loc_id)

    # ── Nearby POIs ───────────────────────────────────────────────────────
    if ablation != "history_only":
        for lid, dist_km in data_loader.get_nearby_pois(query_lon, query_lat, top_n=SPATIAL_TOP_N):
            if lid in seen:
                continue
            meta = data_loader.get_poi_metadata(lid)
            pool.append({
                "loc_id": lid,
                "name": meta["name"],
                "category": meta["category"],
                "dist_km": dist_km,
                "snippet": data_loader.get_poi_snippet(lid, sentences=2),
                "source": "nearby",
                "visit_count": 0,
            })
            seen.add(lid)
            if len(pool) >= MAX_CANDIDATES:
                break

    return pool


def build_quota_pool(
    context: list[dict],
    user_profile: dict,
    similar_users: list[tuple[str, float]],
    target_checkin: dict | None,
    data_loader,
    transitions: dict,
    prior: dict | None = None,
    quota_history: int = QUOTA_HISTORY,
    quota_nearby: int = QUOTA_NEARBY,
    ablation: str = "full",
) -> list[dict]:
    """
    Score each source independently, then take top-K per source.

    Scoring per source:
      history: freq + dow + hb + 0.5×trans
      nearby:  NEARBY_DIST_SCALE / dist_km

    Category-prioritized sorting is disabled when ablation is
    no_intent / no_intent_filtering / no_priors.

    Returns quota_history + quota_nearby candidates
    (less if a source pool is smaller than its quota).
    """
    pool = build_raw_pool(context, user_profile, similar_users, data_loader, ablation=ablation)
    if not pool:
        return []

    # Extract prior signals (only for category-prioritized sort)
    use_prior_categories = ablation not in ("no_intent", "no_intent_filtering", "no_priors", "random_sort")
    likely_categories: set[str] = set()
    if prior and use_prior_categories:
        likely_categories = {c.lower() for c in prior.get("likely_categories", [])}

    last = context[-1]
    temporal_ref = target_checkin or last
    query_dow = temporal_ref.get("weekday", 0)
    query_hb = temporal_ref.get("hour", 0) // HOUR_BUCKET_SIZE
    last_loc_id = last["loc_id"]

    temporal_profile: dict = user_profile.get("temporal_profile", {})
    trans_lookup: dict[int, int] = {dst: cnt for dst, cnt in transitions.get(last_loc_id, [])}

    for c in pool:
        lid = c["loc_id"]
        source = c["source"]
        dist = max(c["dist_km"], 0.01)

        if source == "history":
            t = temporal_profile.get(str(lid), {})
            freq = c["visit_count"]
            dow_cnt = t.get("by_dow", {}).get(str(query_dow), 0)
            hb_cnt = t.get("by_hb", {}).get(str(query_hb), 0)
            trans_cnt = trans_lookup.get(lid, 0)
            c["heuristic_score"] = freq + dow_cnt + hb_cnt + 0.5 * trans_cnt

            parts = []
            if freq > 0:
                parts.append(f"{freq} visit{'s' if freq > 1 else ''}")
            if dow_cnt >= 2:
                parts.append(f"{dow_cnt}× on {_DOW_NAMES[query_dow]}s")
            if hb_cnt >= 2:
                parts.append(f"{hb_cnt}× in {_HB_LABELS[query_hb]} slot")
            if trans_cnt > 0:
                parts.append(f"transition ×{trans_cnt}")
            c["temporal_hint"] = ", ".join(parts) if parts else ""

        else:  # nearby — score by distance only; category priority handled at selection
            c["heuristic_score"] = NEARBY_DIST_SCALE / dist
            c["temporal_hint"] = ""

    if ablation == "random_sort":
        import random

        rng = random.Random(42)

        history_all = [c for c in pool if c["source"] == "history"]
        rng.shuffle(history_all)
        actual_history = min(len(history_all), quota_history)

        nearby_all = [c for c in pool if c["source"] == "nearby"]
        rng.shuffle(nearby_all)
        actual_nearby = min(len(nearby_all), quota_history + quota_nearby - actual_history)

        return history_all[:actual_history] + nearby_all[:actual_nearby]

    # History: category-match first, then heuristic score within each group
    history_all = [c for c in pool if c["source"] == "history"]
    history_match = sorted(
        [c for c in history_all if c["category"].lower() in likely_categories],
        key=lambda x: x["heuristic_score"],
        reverse=True,
    )
    history_rest = sorted(
        [c for c in history_all if c["category"].lower() not in likely_categories],
        key=lambda x: x["heuristic_score"],
        reverse=True,
    )
    history_selected = history_match + history_rest
    actual_history = min(len(history_selected), quota_history)

    # Nearby: category-match first, then distance within each group
    nearby_all = [c for c in pool if c["source"] == "nearby"]
    nearby_match = sorted(
        [c for c in nearby_all if c["category"].lower() in likely_categories],
        key=lambda x: x["dist_km"],
    )
    nearby_rest = sorted(
        [c for c in nearby_all if c["category"].lower() not in likely_categories],
        key=lambda x: x["dist_km"],
    )
    nearby_selected = (nearby_match + nearby_rest)
    actual_nearby = min(len(nearby_selected), quota_history + quota_nearby - actual_history)

    return history_selected[:actual_history] + nearby_selected[:actual_nearby]
