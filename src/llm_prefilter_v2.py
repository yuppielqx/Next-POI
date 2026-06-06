"""
Stage 3b V2: Intent-prior inference + quota-based candidate selection.

Key differences from v1 (llm_prefilter.py):
  - Calls build_quota_pool (candidate_selector_v2) instead of build_scored_pool
  - Prior's likely_categories / likely_specific_places are passed into scoring
  - _filter_by_intent (was dead code in v1) removed
  - Intent prior cache is shared with v1 (same INTENT_CACHE_DIR / format)
"""
from src.candidate_selector_v2 import build_quota_pool
from src.config import (
    FORCED_INCLUDE_N,
    INTENT_CACHE_DIR,
    MAX_CONTEXT_CHECKINS,
    PREFILTER_TOP_N,
    QUOTA_HISTORY,
    QUOTA_NEARBY,
    TOP_K_SIMILAR_USERS,
)
from src.llm_prefilter import (
    _fallback_destination_prior,
    _format_prediction_target,
    _format_similar_patterns,
    _format_trip_context,
    _format_user_time_conditioned_history,
    _infer_destination_prior,
    _prior_to_text,
)
from src.utils import load_json, logger, save_json

# Total candidates = QUOTA_HISTORY + QUOTA_SIMILAR_USER + QUOTA_NEARBY (30)


def select_candidates(
    traj_id: str,
    context: list[dict],
    user_profile: dict,
    similar_users: list[tuple[str, float]],
    target_checkin: dict | None,
    data_loader,
    transitions: dict,
    forced_include_n: int = FORCED_INCLUDE_N,  # kept for API compat, unused in v2
    dry_run: bool = False,
    ablation: str = "full",
    metrics_out: list | None = None,
    quota_history: int | None = None,
    quota_nearby: int | None = None,
) -> list[dict]:
    """
    V2 candidate selection:

      Step 1 (LLM): infer time-conditioned destination prior (cached, same as v1).
                     Skipped when ablation is no_intent / no_priors.
      Step 2 (quota): score each source independently and take per-source top-K.
                      prior category sort disabled for no_intent / no_intent_filtering / no_priors / random_sort.
                      random_sort uses random shuffle instead of heuristic_score.
                      prior text not attached for no_intent / no_priors.

    Returns up to quota_history + quota_nearby candidates.
    Each candidate carries: heuristic_score, temporal_hint, destination_prior (if applicable).
    """
    if quota_history is None:
        quota_history = QUOTA_HISTORY
    if quota_nearby is None:
        quota_nearby = QUOTA_NEARBY

    INTENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Intent cache keyed by ablation: no_profile / no_social have different prompts
    if ablation in ("no_profile", "no_profile_intent"):
        intent_path = INTENT_CACHE_DIR / f"{traj_id}_no_profile.json"
    elif ablation in ("no_social", "no_social_intent"):
        intent_path = INTENT_CACHE_DIR / f"{traj_id}_no_social.json"
    else:
        intent_path = INTENT_CACHE_DIR / f"{traj_id}.json"

    skip_intent_llm = ablation in ("no_intent", "no_priors")
    skip_prior_sort = ablation in ("no_intent", "no_intent_filtering", "no_priors", "random_sort")
    skip_prior_text = ablation in ("no_intent", "no_priors")

    prior: dict | None = None
    if intent_path.exists():
        d = load_json(intent_path)
        if d and d.get("version") == 2 and isinstance(d.get("prior"), dict):
            prior = d.get("prior")

    if dry_run:
        profile_text = user_profile.get("enhanced_profile") or user_profile.get("raw_stats_prompt", "")
        trip_str = _format_trip_context(context, data_loader)
        patterns_str = _format_similar_patterns(similar_users, target_checkin, data_loader)
        target_time_str = _format_prediction_target(target_checkin)
        user_target_history = _format_user_time_conditioned_history(
            user_profile, target_checkin, data_loader
        )
        # Build intent prompt sections matching _infer_destination_prior logic
        remove_profile = ablation in ("no_profile", "no_profile_intent", "no_priors")
        remove_social = ablation in ("no_social", "no_social_intent", "no_priors")
        intent_sections = ["[TIME-CONDITIONED DESTINATION PRIOR PROMPT]"]
        if not remove_profile:
            intent_sections.append(f"## User Profile\n{profile_text[:300]}...")
        intent_sections.append(f"## Recent Trip\n{trip_str}")
        intent_sections.append(f"## Prediction Target Time Prior\n{target_time_str}")
        if not remove_profile:
            intent_sections.append(f"## User's Historical Places Around This Target Time\n{user_target_history}")
        if not remove_social:
            intent_sections.append(f"## Similar Users' Patterns Around The Target Time\n{patterns_str}")
        intent_sections.append("→ Output: JSON destination prior")
        intent_prompt = "\n\n".join(intent_sections)
        logger.info(f"\n{'='*60}\nIntent prompt for {traj_id}:\n{intent_prompt}\n{'='*60}")
        selected = build_quota_pool(
            context, user_profile, similar_users, target_checkin,
            data_loader, transitions,
            prior=(prior or _fallback_destination_prior(user_profile)) if not skip_prior_sort else None,
            ablation=ablation,
            quota_history=quota_history,
            quota_nearby=quota_nearby,
        )
        if not skip_prior_text:
            prior_text = _prior_to_text(prior or _fallback_destination_prior(user_profile))
            for c in selected:
                c["destination_prior"] = prior_text
        return selected

    # ── Step 1: destination prior (cached, shared with v1) ────────────────
    if skip_intent_llm:
        prior = _fallback_destination_prior(user_profile)
    elif prior is None:
        prior = _infer_destination_prior(
            context, user_profile, similar_users, target_checkin, data_loader, ablation=ablation, metrics_out=metrics_out
        )
        save_json(intent_path, {"traj_id": traj_id, "version": 2, "prior": prior})
        logger.info(f"[{traj_id}] Destination prior: {prior.get('summary', 'unknown')}")

    # ── Step 2: quota-based scoring with prior signals ────────────────────
    selected = build_quota_pool(
        context, user_profile, similar_users, target_checkin,
        data_loader, transitions,
        prior=None,
        ablation=ablation,
        quota_history=quota_history,
        quota_nearby=quota_nearby,
    )
    if not skip_prior_text:
        prior_text = _prior_to_text(prior)
        for c in selected:
            c["destination_prior"] = prior_text

    return selected
