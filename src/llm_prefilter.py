"""
Stage 3b: Time-conditioned LLM candidate pre-filtering.

Step 1 (_infer_destination_prior): Given trajectory context + target time
  + similar users' patterns, infer where the user is likely to be at that
  concrete future time.

Step 2 (heuristic): Score the candidate pool using the known target time and
  attach the LLM prior to each candidate for the final ranker.

Results are cached to avoid re-computation on resume.
"""
import json
import time

import src.local_llm as local_llm
from src.candidate_selector import build_raw_pool, build_scored_pool
from src.config import (
    API_MAX_RETRIES,
    API_RETRY_BACKOFF,
    FORCED_INCLUDE_N,
    INTENT_CACHE_DIR,
    INTENT_LLM_MODEL,
    MAX_CONTEXT_CHECKINS,
    PREFILTER_CACHE_DIR,
    PREFILTER_TOP_N,
    TOP_K_SIMILAR_USERS,
)
from src.utils import load_json, logger, movement_direction, save_json, time_of_day_label


# ── Shared prompt helpers ───────────────────────────────────────────────────

def _format_trip_context(context: list[dict], data_loader) -> str:
    checkins = context
    if len(checkins) > MAX_CONTEXT_CHECKINS:
        head = checkins[:2]
        tail = checkins[-(MAX_CONTEXT_CHECKINS - 2):]
        display = head + [None] + tail
    else:
        display = checkins

    lines = []
    counter = 1
    for item in display:
        if item is None:
            lines.append("   … (earlier stops omitted)")
            continue
        name = data_loader.get_poi_name(item["loc_id"])
        cat = data_loader.get_poi_category(item["loc_id"])
        suffix = " ← LAST KNOWN LOCATION" if item is display[-1] else ""
        lines.append(f"{counter}. {item['time']} on {item['date']}: {name} ({cat}){suffix}")
        counter += 1

    direction = movement_direction(context)
    lines.append(f"\nMovement trend: {direction}")
    return "\n".join(lines)


def _time_to_minutes(time_str: str) -> int:
    """Convert 'HH:MM' string to minutes since midnight."""
    h, m = time_str.split(":")
    return int(h) * 60 + int(m)


def _time_diff_minutes(t1: int, t2: int) -> int:
    """Circular difference in minutes between two times-of-day (0–1439)."""
    diff = abs(t1 - t2)
    return min(diff, 1440 - diff)


def _format_similar_patterns(
    similar_users: list[tuple[str, float]],
    target_checkin: dict | None,
    data_loader,
) -> str:
    """
    For each similar user, find check-ins whose time-of-day is within 30 minutes
    of the target time and show the local transition around that moment.
    """
    if not target_checkin:
        return "No target-time examples available."
    target_minutes = _time_to_minutes(target_checkin.get("time", "00:00"))

    lines = []
    for uid, score in similar_users[:TOP_K_SIMILAR_USERS]:
        trips = data_loader.get_user_train_val_trips(uid)
        examples = []
        for trip in trips:
            for i, c in enumerate(trip[:-1]):
                if _time_diff_minutes(_time_to_minutes(c["time"]), target_minutes) > 30:
                    continue
                cur_name = data_loader.get_poi_name(c["loc_id"])
                cur_cat = data_loader.get_poi_category(c["loc_id"])
                next_c = trip[i + 1]
                next_name = data_loader.get_poi_name(next_c["loc_id"])
                next_cat = data_loader.get_poi_category(next_c["loc_id"])
                examples.append(
                    f"around {target_checkin.get('time', '')}, was at {cur_name} ({cur_cat}), "
                    f"then visited {next_name} ({next_cat})"
                )
                if len(examples) >= 2:
                    break
            if len(examples) >= 2:
                break
        if examples:
            lines.append(f"- Similar user (similarity {score:.2f}): {'; '.join(examples)}")
    return "\n".join(lines) if lines else "No matching patterns found in similar users."


def _format_prediction_target(target_checkin: dict | None) -> str:
    """Describe the held-out next-checkin time without exposing the true location."""
    if not target_checkin:
        return "Unknown target time."

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = weekday_names[target_checkin.get("weekday", 0) % 7]
    hour = target_checkin.get("hour", 0)
    return (
        f"The next visit to predict happens on {target_checkin.get('date', 'unknown date')} "
        f"({weekday}) at {target_checkin.get('time', 'unknown time')} during the "
        f"{time_of_day_label(hour)}. Use this as a prior, but do not assume the destination."
    )


def _format_user_time_conditioned_history(
    user_profile: dict,
    target_checkin: dict | None,
    data_loader,
    limit: int = 8,
) -> str:
    """Summarize the user's historically strongest places around the target time."""
    if not target_checkin:
        return "No target-time user history available."

    temporal_profile: dict = user_profile.get("temporal_profile", {})
    target_dow = str(target_checkin.get("weekday", 0))
    target_hb = str(target_checkin.get("hour", 0) // 3)
    scored = []
    for loc_id_str, stats in temporal_profile.items():
        total = int(stats.get("total", 0))
        dow_cnt = int(stats.get("by_dow", {}).get(target_dow, 0))
        hb_cnt = int(stats.get("by_hb", {}).get(target_hb, 0))
        score = 2 * dow_cnt + hb_cnt + 0.2 * total
        if score <= 0:
            continue
        loc_id = int(loc_id_str)
        scored.append((score, total, dow_cnt, hb_cnt, loc_id))

    scored.sort(reverse=True)
    lines = []
    for _, total, dow_cnt, hb_cnt, loc_id in scored[:limit]:
        name = data_loader.get_poi_name(loc_id)
        cat = data_loader.get_poi_category(loc_id)
        parts = [f"{total} total visits"]
        if dow_cnt:
            parts.append(f"{dow_cnt} on same weekday")
        if hb_cnt:
            parts.append(f"{hb_cnt} in same time slot")
        lines.append(f"- {name} ({cat}): {', '.join(parts)}")
    return "\n".join(lines) if lines else "No strong user history exactly matching the target time."


def _fallback_destination_prior(user_profile: dict) -> dict:
    top_categories = user_profile.get("top_categories", [])[:3]
    return {
        "summary": "Likely revisiting a familiar place near the target time.",
        "revisit_probability": "medium",
        "likely_area": "unknown",
        "likely_categories": top_categories,
        "likely_specific_places": [],
        "movement_type": "return_to_frequent_place",
        "rationale": "Fallback prior based on the user's repeated historical categories.",
    }


def _extract_json_object(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def _parse_destination_prior(text: str | None, user_profile: dict) -> dict:
    if not text:
        return _fallback_destination_prior(user_profile)

    json_text = _extract_json_object(text)
    if not json_text:
        return _fallback_destination_prior(user_profile)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return _fallback_destination_prior(user_profile)

    prior = _fallback_destination_prior(user_profile)
    for key in ["summary", "revisit_probability", "likely_area", "movement_type", "rationale"]:
        if isinstance(data.get(key), str) and data[key].strip():
            prior[key] = data[key].strip()

    for key in ["likely_categories", "likely_specific_places"]:
        value = data.get(key, [])
        if isinstance(value, list):
            prior[key] = [str(item).strip() for item in value if str(item).strip()][:5]

    return prior


def _prior_to_text(prior: dict) -> str:
    categories = ", ".join(prior.get("likely_categories", [])) or "unknown"
    places = ", ".join(prior.get("likely_specific_places", [])) or "none"
    return (
        f"Summary: {prior.get('summary', 'unknown')}\n"
        f"Revisit probability: {prior.get('revisit_probability', 'unknown')}\n"
        f"Likely area: {prior.get('likely_area', 'unknown')}\n"
        f"Likely categories: {categories}\n"
        f"Likely specific places: {places}\n"
        f"Movement type: {prior.get('movement_type', 'unknown')}\n"
        f"Rationale: {prior.get('rationale', 'unknown')}"
    )


def _llm_call(
    model: str, prompt: str, max_tokens: int = 400, system: str | None = None,
) -> str | None:
    """Make one LLM call with retry; return response text or None on failure."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    for attempt in range(API_MAX_RETRIES):
        try:
            return local_llm.chat_completion(model, messages, max_new_tokens=max_tokens)
        except Exception as e:
            wait = API_RETRY_BACKOFF * (2 ** attempt)
            logger.warning(f"Inference error (attempt {attempt + 1}): {e}, waiting {wait}s")
            time.sleep(wait)
    return None


# ── Step 1: Time-conditioned destination prior ─────────────────────────────

def _infer_destination_prior(
    context: list[dict],
    user_profile: dict,
    similar_users: list[tuple[str, float]],
    target_checkin: dict | None,
    data_loader,
) -> dict:
    trip_str = _format_trip_context(context, data_loader)
    patterns_str = _format_similar_patterns(similar_users, target_checkin, data_loader)

    system = (
        "You predict where a New York City user is likely to be at a known future time. "
        "Return only JSON."
    )

    profile_text = user_profile.get("enhanced_profile") or user_profile.get("raw_stats_prompt", "")
    target_time_str = _format_prediction_target(target_checkin)
    user_target_history = _format_user_time_conditioned_history(user_profile, target_checkin, data_loader)

    prompt = f"""## User Profile
{profile_text}

## Recent Trip (latest last)
{trip_str}

## Prediction Target Time Prior
{target_time_str}

## User's Historical Places Around This Target Time
{user_target_history}

## What Similar Users Did Around The Target Time (±30 min)
{patterns_str}

Based on this trajectory and the known target time, infer where the user is most likely to be at that moment.
Return JSON with exactly these keys:
- summary
- revisit_probability
- likely_area
- likely_categories
- likely_specific_places
- movement_type
- rationale

Rules:
- likely_categories and likely_specific_places must be arrays
- likely_specific_places may contain exact venue names from the user's own history when appropriate
- movement_type must be one of: return_to_last, return_to_frequent_place, nearby_new_place, long_distance_jump
- Do not include markdown fences or extra text"""

    text = _llm_call(INTENT_LLM_MODEL, prompt, max_tokens=300, system=system)
    prior = _parse_destination_prior(text, user_profile)
    if not text:
        logger.warning("Destination-prior inference failed; using fallback prior.")
    return prior


# ── Step 2: Intent-Based Candidate Filtering ────────────────────────────────

def _format_pool_for_prompt(raw_pool: list[dict]) -> str:
    lines = []
    for i, c in enumerate(raw_pool, 1):
        source_tag = (
            f"history ({c['visit_count']} visits)" if c["source"] == "history"
            else "similar user"
        )
        lines.append(
            f"[{i}] {c['name']} | {c['category']} | {c['dist_km']:.2f} km | {source_tag}"
        )
    return "\n".join(lines)


def _filter_by_intent(
    intent: str,
    raw_pool: list[dict],
    top_n: int,
) -> list[dict]:
    pool_str = _format_pool_for_prompt(raw_pool)
    n = len(raw_pool)

    system = (
        "You are a location filtering assistant. "
        "Given a user's inferred intent and a list of candidate POIs, "
        "select the ones most relevant to the intent that are also geographically close."
    )

    prompt = f"""User's current intent: "{intent}"

From the following candidate locations, select the {top_n} that best match this intent and are geographically close to the last known location.

## Candidate Locations ({n} total)
{pool_str}

Output exactly one line: SELECTED: <{top_n} space-separated 1-based indices>
Example: SELECTED: 3 7 12 15 ..."""

    text = _llm_call(PREFILTER_LLM_MODEL, prompt, max_tokens=200, system=system)
    selected_ids: list[int] = []
    if text:
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("SELECTED:"):
                parts = line[len("SELECTED:"):].strip().replace(",", " ").split()
                for p in parts:
                    try:
                        idx = int(p)
                        if 1 <= idx <= n:
                            selected_ids.append(idx - 1)  # convert to 0-based
                    except ValueError:
                        continue
                break

    # Deduplicate preserving order
    seen: set[int] = set()
    deduped: list[int] = []
    for idx in selected_ids:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)

    selected = [raw_pool[i] for i in deduped[:top_n]]

    # Pad with nearest-by-dist if LLM returned fewer than top_n
    if len(selected) < top_n:
        selected_set = {raw_pool[i]["loc_id"] for i in deduped[:top_n]}
        remaining = sorted(
            [c for c in raw_pool if c["loc_id"] not in selected_set],
            key=lambda x: x["dist_km"],
        )
        selected += remaining[: top_n - len(selected)]
        logger.warning(
            f"Intent filter returned {len(deduped)} indices; padded to {len(selected)} by distance."
        )

    return selected


# ── Public entry point ──────────────────────────────────────────────────────

def select_candidates(
    traj_id: str,
    context: list[dict],
    user_profile: dict,
    similar_users: list[tuple[str, float]],
    target_checkin: dict | None,
    data_loader,
    transitions: dict,
    forced_include_n: int = FORCED_INCLUDE_N,
    dry_run: bool = False,
) -> list[dict]:
    """
    Candidate selection: target-time heuristic scoring + LLM destination prior.

    Step 1 (LLM): Infer a time-conditioned destination prior from trajectory
                  + target time + similar-user patterns.
    Step 2 (heuristic): Score candidates by freq + day-of-week + hour-bucket
                        + transition signal; return top PREFILTER_TOP_N.

    Returns up to PREFILTER_TOP_N candidates as list of dicts:
    {loc_id, name, category, dist_km, snippet, source, visit_count,
     heuristic_score, temporal_hint, destination_prior}
    """
    # ── Check prior cache ────────────────────────────────────────────────
    INTENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    intent_path = INTENT_CACHE_DIR / f"{traj_id}.json"
    prior: dict | None = None
    if intent_path.exists():
        d = load_json(intent_path)
        if d and d.get("version") == 2 and isinstance(d.get("prior"), dict):
            prior = d.get("prior")

    if dry_run:
        # Show prior prompt and heuristic pool, no LLM call
        profile_text = user_profile.get("enhanced_profile") or user_profile.get("raw_stats_prompt", "")
        trip_str = _format_trip_context(context, data_loader)
        patterns_str = _format_similar_patterns(similar_users, target_checkin, data_loader)
        target_time_str = _format_prediction_target(target_checkin)
        user_target_history = _format_user_time_conditioned_history(user_profile, target_checkin, data_loader)
        intent_prompt = f"""[TIME-CONDITIONED DESTINATION PRIOR PROMPT]
[system] You predict where a New York City user is likely to be at a known future time. Return only JSON.
## User Profile
{profile_text[:300]}...
## Recent Trip
{trip_str}
## Prediction Target Time Prior
{target_time_str}
## User's Historical Places Around This Target Time
{user_target_history}
## Similar Users' Patterns Around The Target Time
{patterns_str}
→ Output: JSON destination prior"""
        logger.info(f"\n{'='*60}\nIntent prompt for {traj_id}:\n{intent_prompt}\n{'='*60}")
        selected = build_scored_pool(
            context,
            user_profile,
            similar_users,
            target_checkin,
            data_loader,
            transitions,
            top_n=PREFILTER_TOP_N,
            forced_include_n=forced_include_n,
        )
        prior_text = _prior_to_text(prior or _fallback_destination_prior(user_profile))
        for c in selected:
            c["destination_prior"] = prior_text
        return selected

    # ── Step 1: destination prior (cached) ───────────────────────────────
    if prior is None:
        prior = _infer_destination_prior(context, user_profile, similar_users, target_checkin, data_loader)
        save_json(intent_path, {"traj_id": traj_id, "version": 2, "prior": prior})
        logger.info(f"[{traj_id}] Destination prior: {prior.get('summary', 'unknown')}")

    # ── Step 2: heuristic temporal scoring ───────────────────────────────
    selected = build_scored_pool(
        context,
        user_profile,
        similar_users,
        target_checkin,
        data_loader,
        transitions,
        top_n=PREFILTER_TOP_N,
        forced_include_n=forced_include_n,
    )
    prior_text = _prior_to_text(prior)
    for c in selected:
        c["destination_prior"] = prior_text

    return selected
