"""
Stage 3c: LLM-based next POI prediction agent.

Uses gpt-5.4-mini to rank candidate locations given:
  - User's enhanced profile
  - Current trip context
  - Similar users' patterns
  - Candidate POI descriptions
"""
import difflib
import time

import src.local_llm as local_llm
from src.config import (
    API_MAX_RETRIES,
    API_RETRY_BACKOFF,
    MAX_CONTEXT_CHECKINS,
    PREDICTION_LLM_MODEL,
)
from src.utils import logger, movement_direction, time_of_day_label


# ── Prompt assembly ────────────────────────────────────────────────────────

def _format_context(context: list[dict], data_loader) -> str:
    """Format trip context checkins into readable numbered list."""
    checkins = context
    # Truncate long trips: keep first 2 + last (MAX-2) checkins
    if len(checkins) > MAX_CONTEXT_CHECKINS:
        head = checkins[:2]
        tail = checkins[-(MAX_CONTEXT_CHECKINS - 2):]
        display = head + [None] + tail  # None = ellipsis marker
    else:
        display = checkins

    lines = []
    counter = 1
    for item in display:
        if item is None:
            lines.append(f"   … (earlier stops omitted)")
            continue
        name = data_loader.get_poi_name(item["loc_id"])
        cat = data_loader.get_poi_category(item["loc_id"])
        suffix = " ← LAST KNOWN LOCATION" if item is display[-1] else ""
        lines.append(f"{counter}. {item['time']} on {item['date']}: {name} ({cat}){suffix}")
        counter += 1

    direction = movement_direction(context)
    lines.append(f"\nMovement trend: {direction}")
    return "\n".join(lines)


def _format_similar_users_patterns(
    similar_users: list[tuple[str, float]],
    target_checkin: dict | None,
    data_loader,
) -> str:
    """Show what similar users visited around the target time to ground the final choice."""
    if not similar_users or not target_checkin:
        return "No similar user data available."

    target_minutes = int(target_checkin.get("hour", 0)) * 60 + int(target_checkin.get("time", "00:00").split(":")[1])
    lines = []
    for uid, score in similar_users[:3]:
        trips = data_loader.get_user_train_val_trips(uid)
        examples = []
        for trip in trips:
            for i, c in enumerate(trip[:-1]):
                cur_minutes = int(c.get("hour", 0)) * 60 + int(c.get("time", "00:00").split(":")[1])
                diff = abs(cur_minutes - target_minutes)
                diff = min(diff, 1440 - diff)
                if diff > 30:
                    continue
                next_c = trip[i + 1]
                cur_name = data_loader.get_poi_name(c["loc_id"])
                cur_cat = data_loader.get_poi_category(c["loc_id"])
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


def _format_candidates(candidates: list[dict]) -> str:
    """Format candidate list for the LLM prompt, including temporal signals."""
    lines = []
    for i, c in enumerate(candidates, 1):
        hint = c.get("temporal_hint", "")
        history_line = f"    Your history: {hint}" if hint else "    Your history: no previous visits"
        source = c.get("source", "unknown")
        source_label = {
            "history": "user history",
            "similar_user": "similar users",
            "nearby": "nearby search",
        }.get(source, source)
        if c.get("forced_nearby"):
            source_label += " (forced nearby keep)"
        extra_lines = []
        extra_lines.append(f"    Source: {source_label}")
        extra_lines.append(f"    Context: {c['snippet']}")
        details = "\n".join(extra_lines)
        lines.append(
            f"[{i}] {c['name']}\n"
            f"    Category: {c['category']} | Distance: {c['dist_km']:.2f} km from last location\n"
            f"{history_line}\n"
            f"{details}"
        )
    return "\n\n".join(lines)


def _format_prediction_target(target_checkin: dict | None) -> str:
    """Describe the held-out next-checkin time without revealing its location."""
    if not target_checkin:
        return "Unknown target time."

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = weekday_names[target_checkin.get("weekday", 0) % 7]
    hour = target_checkin.get("hour", 0)
    return (
        f"The next visit to predict happens on {target_checkin.get('date', 'unknown date')} "
        f"({weekday}) at {target_checkin.get('time', 'unknown time')} during the "
        f"{time_of_day_label(hour)}. Treat this as a strong temporal prior while choosing among candidates."
    )


def _build_prompt(
    user_profile: dict,
    context: list[dict],
    similar_users: list[tuple[str, float]],
    candidates: list[dict],
    target_checkin: dict | None,
    data_loader,
) -> str:
    n_candidates = len(candidates)

    # Extract time-conditioned destination prior from candidates (set by select_candidates)
    destination_prior = candidates[0].get("destination_prior", "") if candidates else ""

    task_section = f"""You are a location prediction expert. A user is traveling in New York City.
Given their movement history, behavioral profile, and patterns from similar users, predict which location they will visit next.

Output format: A ranked list of exactly {min(10, n_candidates)} entries, numbered 1 (most likely) to {min(10, n_candidates)} (least likely). Each line must follow this format exactly:
<number>. <exact location name> | <one sentence reason>
Use the exact names as given in the candidate list. Output ONLY the numbered list, nothing else."""

    profile_section = f"""## User Profile
{user_profile.get('enhanced_profile', user_profile.get('raw_stats_prompt', 'No profile available.'))}"""

    context_section = f"""## Current Trip
{_format_context(context, data_loader)}"""

    similar_section = f"""## Patterns from Similar Users Around The Target Time
{_format_similar_users_patterns(similar_users, target_checkin, data_loader)}"""

    prior_section = (
        f"""## Time-Conditioned Destination Prior\n{destination_prior}"""
        if destination_prior else ""
    )

    target_section = f"""## Prediction Target Time Prior
{_format_prediction_target(target_checkin)}"""

    candidate_section = f"""## Candidate Locations (select from these {n_candidates} options only)
{_format_candidates(candidates)}"""

    sections = [task_section, profile_section, context_section, target_section, similar_section]
    if prior_section:
        sections.append(prior_section)
    sections.append(candidate_section)
    return "\n\n".join(sections)


# ── Output parsing ─────────────────────────────────────────────────────────

def _parse_ranked_list(response_text: str, candidates: list[dict]) -> tuple[list[int], dict[int, str]]:
    """
    Parse LLM output into a ranked list of loc_ids and a dict of explanations.
    Expected line format: "1. Name | reason"
    Matches names exactly then fuzzy.
    Returns (ranked_ids, {loc_id: reason})
    """
    candidate_names = [c["name"] for c in candidates]
    name_to_id = {c["name"]: c["loc_id"] for c in candidates}

    ranked_ids = []
    explanations: dict[int, str] = {}

    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove leading numbering
        clean = line.lstrip("0123456789.-) ").strip()
        if not clean:
            continue

        # Split name and reason on "|"
        if "|" in clean:
            name_part, reason = clean.split("|", 1)
            name_part = name_part.strip()
            reason = reason.strip()
        else:
            name_part = clean
            reason = ""

        # Exact match
        if name_part in name_to_id:
            lid = name_to_id[name_part]
            if lid not in ranked_ids:
                ranked_ids.append(lid)
                explanations[lid] = reason
            continue

        # Fuzzy match
        match = difflib.get_close_matches(name_part, candidate_names, n=1, cutoff=0.6)
        if match:
            lid = name_to_id[match[0]]
            if lid not in ranked_ids:
                ranked_ids.append(lid)
                explanations[lid] = reason

    return ranked_ids, explanations


# ── Main prediction function ───────────────────────────────────────────────

def predict_next_poi(
    traj_id: str,
    context: list[dict],
    user_profile: dict,
    similar_users: list[tuple[str, float]],
    candidates: list[dict],
    target_checkin: dict | None,
    data_loader,
    dry_run: bool = False,
) -> dict:
    """
    Predict the next POI for a test trajectory.

    Returns:
      {traj_id, ranked_loc_ids, prompt (if dry_run), fallback}
    """
    if not candidates:
        return {"traj_id": traj_id, "ranked_loc_ids": [], "fallback": True}

    prompt = _build_prompt(user_profile, context, similar_users, candidates, target_checkin, data_loader)

    if dry_run:
        return {"traj_id": traj_id, "prompt": prompt, "ranked_loc_ids": [], "fallback": False}

    ranked_ids = []
    explanations: dict[int, str] = {}
    used_fallback = False

    for attempt in range(API_MAX_RETRIES):
        try:
            text = local_llm.chat_completion(
                model=PREDICTION_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=600,
            )
            if text:
                ranked_ids, explanations = _parse_ranked_list(text, candidates)
            if ranked_ids:
                break
        except Exception as e:
            wait = API_RETRY_BACKOFF * (2 ** attempt)
            logger.warning(f"Inference error for {traj_id} (attempt {attempt+1}): {e}, waiting {wait}s")
            time.sleep(wait)

    # Fallback: return candidates ordered by dist score
    if not ranked_ids:
        used_fallback = True
        ranked_ids = [c["loc_id"] for c in sorted(candidates, key=lambda x: x["dist_km"])]

    return {
        "traj_id": traj_id,
        "ranked_loc_ids": ranked_ids[:10],
        "explanations": {str(k): v for k, v in explanations.items()},
        "fallback": used_fallback,
    }
