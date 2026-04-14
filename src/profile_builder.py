"""
Stage 1: Build LLM-enhanced user profiles.

For each user, computes statistics from their train+val trajectories and
calls gpt-5.4 to generate a rich narrative profile. Results are cached
individually so the process is fully resume-safe.
"""
import concurrent.futures
import json
import time
from pathlib import Path

import geohash2

import src.local_llm as local_llm
from src.config import (
    EMBEDDING_MODEL,
    GEOHASH_PRECISION,
    HOUR_BUCKET_SIZE,
    PROFILE_LLM_MODEL,
    PROFILE_MAX_CHECKINS,
    PROFILE_MAX_TRIPS,
    PROFILES_CACHE_DIR,
    API_MAX_RETRIES,
    API_RETRY_BACKOFF,
)
from src.utils import Progress, logger, save_json, load_json


def _build_visit_history(trips: list[list[dict]], data_loader) -> str:
    """Format train+val trips as a readable visit table."""
    lines = []
    trips_to_use = trips[-PROFILE_MAX_TRIPS:] if len(trips) > PROFILE_MAX_TRIPS else trips
    for trip in trips_to_use:
        checkins = trip[:PROFILE_MAX_CHECKINS]
        for c in checkins:
            name = data_loader.get_poi_name(c["loc_id"])
            cat = data_loader.get_poi_category(c["loc_id"])
            lines.append(f"{c['date']} {c['time']} | {name} | {cat}")
    return "\n".join(lines)


_SYSTEM_PROMPT = "You are a mobility analyst. Based on the following data about a Foursquare user in New York City, write a comprehensive behavioral profile (2-3 paragraphs, approximately 200 words)."


def _profile_prompt(stats_prompt: str, visit_history: str, n_trips: int) -> tuple[str, str]:
    user_prompt = f"""## Statistical Summary
{stats_prompt}

## Visit History ({n_trips} trips, train+validation set)
{visit_history}

## Instructions
Synthesize the above into a narrative profile covering:
1. TEMPORAL: Preferred hours and days (morning commuter? weekend night-owl?)
2. SPATIAL: Which neighborhoods does this user frequent? How wide is their range?
3. CATEGORIES: Dominant venue types (bars, coffee shops, gyms, etc.)
4. MOBILITY STYLE: Do they cluster visits or spread across the city?
5. SIGNATURES: Any distinctive recurring patterns (e.g., always visits coffee shop before work)

Write in third person, present tense. Be specific — reference actual venue names and neighborhoods where possible. Do NOT include bullet points or headers; write flowing paragraphs."""
    return _SYSTEM_PROMPT, user_prompt


def _compute_temporal_profile(trips: list[list[dict]]) -> dict:
    """
    Build per-loc temporal visit distribution from train+val trips.

    Returns {str(loc_id): {"total": int, "by_dow": {str(0-6): int}, "by_hb": {str(0-7): int}}}
    where by_dow keys are weekday strings (0=Mon … 6=Sun) and by_hb keys are
    hour-bucket strings (hour // HOUR_BUCKET_SIZE).
    """
    temporal: dict[str, dict] = {}
    for trip in trips:
        for c in trip:
            key = str(c["loc_id"])
            if key not in temporal:
                temporal[key] = {"total": 0, "by_dow": {}, "by_hb": {}}
            temporal[key]["total"] += 1
            dow = str(c["weekday"])
            temporal[key]["by_dow"][dow] = temporal[key]["by_dow"].get(dow, 0) + 1
            hb = str(c["hour"] // HOUR_BUCKET_SIZE)
            temporal[key]["by_hb"][hb] = temporal[key]["by_hb"].get(hb, 0) + 1
    return temporal


def build_profile(user_id: str, data_loader) -> dict:
    """
    Build and cache an enhanced profile for one user.
    Returns the cached profile dict (whether newly built or loaded from cache).
    """
    PROFILES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PROFILES_CACHE_DIR / f"{user_id}.json"

    cached = load_json(cache_path)
    if cached is not None:
        return cached

    # Gather data
    user_stats = data_loader.compute_user_stats(user_id)
    stats_prompt = user_stats["stats_prompt"]
    trips = data_loader.get_user_train_val_trips(user_id)
    n_trips = len(trips)

    # Collect visited loc_ids (with duplicates = visit counts) and geohashes
    visited_loc_ids: list[int] = []  # includes duplicates for frequency counting
    visited_geohashes: set[str] = set()
    for trip in trips:
        for c in trip:
            lid = c["loc_id"]
            visited_loc_ids.append(lid)  # keep duplicates
            gh = geohash2.encode(c["lat"], c["lon"], precision=GEOHASH_PRECISION)
            visited_geohashes.add(gh)

    # Top categories computed from trip data
    top_categories = user_stats["top_categories"]

    visit_history = _build_visit_history(trips, data_loader)
    system_prompt, user_prompt = _profile_prompt(stats_prompt, visit_history, n_trips)

    # Call LLM with retry
    enhanced_profile = ""
    for attempt in range(API_MAX_RETRIES):
        try:
            text = local_llm.chat_completion(
                model=PROFILE_LLM_MODEL,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                ],
            )
            if text:
                enhanced_profile = text
                break
        except Exception as e:
            wait = API_RETRY_BACKOFF * (2 ** attempt)
            logger.warning(f"Inference error for user {user_id} (attempt {attempt+1}): {e}, waiting {wait}s")
            time.sleep(wait)
    if not enhanced_profile:
        enhanced_profile = stats_prompt  # fallback to stats

    profile = {
        "user_id": user_id,
        "enhanced_profile": enhanced_profile,
        "raw_stats_prompt": stats_prompt,
        "num_trips": n_trips,
        "top_categories": top_categories,
        "visited_loc_ids": visited_loc_ids,
        "visited_geohashes": sorted(visited_geohashes),
        "temporal_profile": _compute_temporal_profile(trips),
    }
    save_json(cache_path, profile)
    return profile


def build_all_profiles(data_loader, force: bool = False, workers: int = 1) -> dict[str, dict]:
    """
    Build enhanced profiles for all users. Resume-safe: skips existing cache files.
    Returns mapping user_id → profile dict.
    """
    PROFILES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_users = data_loader.get_all_user_ids()

    if force:
        # Clear existing cache
        for f in PROFILES_CACHE_DIR.glob("*.json"):
            f.unlink()

    to_build = [
        uid for uid in all_users
        if not (PROFILES_CACHE_DIR / f"{uid}.json").exists()
    ]
    already_done = len(all_users) - len(to_build)
    logger.info(
        f"Profile building: {already_done} cached, {len(to_build)} to build "
        f"(total {len(all_users)})"
    )

    progress = Progress(len(to_build), "Profiles")
    profiles: dict[str, dict] = {}

    # Load already-cached profiles
    for uid in all_users:
        p = load_json(PROFILES_CACHE_DIR / f"{uid}.json")
        if p is not None:
            profiles[uid] = p

    # Build missing ones
    if workers > 1 and len(to_build) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(build_profile, uid, data_loader): uid for uid in to_build}
            for future in concurrent.futures.as_completed(futures):
                uid = futures[future]
                profiles[uid] = future.result()
                progress.step()
    else:
        for uid in to_build:
            profiles[uid] = build_profile(uid, data_loader)
            progress.step()

    logger.info(f"All {len(profiles)} profiles ready.")
    return profiles


def load_all_profiles() -> dict[str, dict]:
    """Load all cached profiles without building new ones."""
    profiles = {}
    for path in PROFILES_CACHE_DIR.glob("*.json"):
        data = load_json(path)
        if data:
            profiles[data["user_id"]] = data
    return profiles


def migrate_temporal_profiles(data_loader) -> int:
    """
    Add temporal_profile to existing cached profile JSONs that lack it.
    Reads each cached profile, computes temporal_profile from raw trip data
    (no LLM calls), and overwrites the file. Returns number of profiles updated.
    """
    updated = 0
    for path in sorted(PROFILES_CACHE_DIR.glob("*.json")):
        profile = load_json(path)
        if profile is None:
            continue
        if "temporal_profile" in profile:
            continue  # already migrated
        uid = profile.get("user_id")
        if not uid:
            continue
        trips = data_loader.get_user_train_val_trips(uid)
        profile["temporal_profile"] = _compute_temporal_profile(trips)
        save_json(path, profile)
        updated += 1
    return updated
