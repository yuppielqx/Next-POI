"""
Stage 3a: Build and heuristically score the candidate POI pool.

Collects candidates from:
  1. User's own historical POIs (all unique visits)
  2. Similar users' POIs (not already in user history)
  3. Nearby POIs around the last known location

Scoring formula (build_scored_pool):
  score = freq + 2×same_dow + 1×same_hour_bucket + 0.5×transition_count

Returns a flat list sorted by heuristic score descending.
"""
from collections import Counter

from src.config import (
    FORCED_INCLUDE_N,
    HOUR_BUCKET_SIZE,
    MAX_CANDIDATES,
    SPATIAL_TOP_N,
    TOP_K_SIMILAR_USERS,
    RAW_POOL_SIMILAR_TOP_N,
)
from src.utils import haversine

# Hour-bucket label strings for temporal hints
_HB_LABELS = [
    "midnight–2am", "3–5am", "6–8am", "9–11am",
    "noon–2pm", "3–5pm", "6–8pm", "9–11pm",
]
_DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def build_raw_pool(
    context: list[dict],
    user_profile: dict,
    similar_users: list[tuple[str, float]],
    data_loader,
) -> list[dict]:
    """
    Return raw candidate pool as list of dicts:
    {loc_id, name, category, dist_km, snippet, source, visit_count}

    source: "history" | "similar_user" | "nearby"
    visit_count: number of times user visited (0 for non-history POIs)
    """
    if not context:
        return []

    last = context[-1]
    query_lon, query_lat = last["lon"], last["lat"]

    visited_loc_ids: list[int] = user_profile.get("visited_loc_ids", [])
    visit_freq = Counter(visited_loc_ids)
    visited_set = set(visited_loc_ids)

    # ── Distance helper ───────────────────────────────────────────────────
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

    # ── Similar users' POIs ───────────────────────────────────────────────
    sim_counter: Counter = Counter()
    for uid, sim_score in similar_users[:TOP_K_SIMILAR_USERS]:
        for trip in data_loader.get_user_train_val_trips(uid):#相似user的所有历史轨迹
            for c in trip:
                lid = c["loc_id"]
                if lid not in visited_set:
                    sim_counter[lid] += sim_score

    for lid, _ in sim_counter.most_common(RAW_POOL_SIMILAR_TOP_N):
        if lid in seen:
            continue
        meta = data_loader.get_poi_metadata(lid)
        pool.append({
            "loc_id": lid,
            "name": meta["name"],
            "category": meta["category"],
            "dist_km": get_dist(lid),
            "snippet": data_loader.get_poi_snippet(lid, sentences=2),
            "source": "similar_user",
            "visit_count": 0,
        })
        seen.add(lid)

    # ── Nearby POIs around the last known location ───────────────────────
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


def build_scored_pool(
    context: list[dict],
    user_profile: dict,
    similar_users: list[tuple[str, float]],
    target_checkin: dict | None,
    data_loader,
    transitions: dict,
    top_n: int = 30,
    forced_include_n: int = FORCED_INCLUDE_N,
) -> list[dict]:
    """
    Build raw pool then score and sort candidates by temporal match.

    Scoring:
      score = freq + 2×same_dow + 1×same_hour_bucket + 0.5×transition_count

    Temporal reference: target_checkin weekday and hour when available, falling
    back to the last known location time otherwise.
    Returns the top_n candidates sorted by score descending, each with added
    fields: heuristic_score, temporal_hint (human-readable match description).
    """
    pool = build_raw_pool(context, user_profile, similar_users, data_loader)
    if not pool:
        return []

    last = context[-1]
    temporal_ref = target_checkin or last
    query_dow = temporal_ref.get("weekday", 0)
    query_hb = temporal_ref.get("hour", 0) // HOUR_BUCKET_SIZE
    last_loc_id = last["loc_id"]

    temporal_profile: dict = user_profile.get("temporal_profile", {})
    # transitions: {int → [(int, int)]} — convert to fast lookup dict
    trans_lookup: dict[int, int] = {dst: cnt for dst, cnt in transitions.get(last_loc_id, [])}

    for c in pool:
        lid = c["loc_id"]
        t = temporal_profile.get(str(lid), {})
        freq = c["visit_count"]
        dow_cnt = t.get("by_dow", {}).get(str(query_dow), 0)
        hb_cnt = t.get("by_hb", {}).get(str(query_hb), 0)
        trans_cnt = trans_lookup.get(lid, 0)

        c["heuristic_score"] = freq + 2 * dow_cnt + hb_cnt + 0.5 * trans_cnt

        # Build a readable hint for the LLM prompt
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

    pool.sort(key=lambda x: (x["heuristic_score"], -x["dist_km"]), reverse=True)
    selected = pool[:top_n]

    # Force a few nearest POIs into the shortlist so that plausible nearby-new
    # destinations are not discarded purely for lacking visit history.
    forced_nearby = sorted(pool, key=lambda x: x["dist_km"])[:forced_include_n]
    selected_ids = {c["loc_id"] for c in selected}
    if len(selected) >= top_n:
        selected.sort(key=lambda x: (x["heuristic_score"], -x["dist_km"]))
    for cand in forced_nearby:
        if cand["loc_id"] in selected_ids:
            continue
        if len(selected) >= top_n:
            removed = selected.pop(0)
            selected_ids.discard(removed["loc_id"])
        cand = dict(cand)
        cand["forced_nearby"] = True
        selected.append(cand)
        selected_ids.add(cand["loc_id"])

    selected.sort(key=lambda x: (x["heuristic_score"], -x["dist_km"]), reverse=True)
    return selected[:top_n]
