"""
Stage 3 V2: Predict the next POI using quota-based candidate selection.

Differences from predict.py (v1):
  - Uses src.candidate_selector_v2 and src.llm_prefilter_v2
  - Writes predictions to cache/<dataset>/predictions_v2/ (separate from v1)
  - Intent prior cache is shared with v1 (no re-inference needed)

Usage:
  python predict_v2.py [--traj-id TRAJID]
                       [--dry-run]
                       [--workers N]
                       [--recompute]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.candidate_selector_v2 import build_raw_pool
from src.config import (
    CACHE_DIR,
    DATASET_TAG,
    DATA_DIR,
    INTENT_CACHE_DIR,
    POOLS_V2_CACHE_DIR,
    PREDICTIONS_V2_CACHE_DIR,
    PROFILES_CACHE_DIR,
    SIMILARITY_CACHE,
    TRANSITIONS_CACHE,
)
from src.data_loader import DataLoader
from src.user_similarity import get_similar_users
from src.utils import Progress, load_json, logger, save_json

ABLATION_VARIANTS = {
    "full",
    "no_profile",
    "no_profile_intent",
    "no_profile_rank",
    "no_social",
    "no_social_rank",
    "no_social_intent",
    "no_intent",
    "no_priors",
    "no_intent_filtering",
    "random_sort",
    "history_only",
    "spatial_only",
}

try:
    from src.llm_agent import predict_next_poi
    from src.llm_prefilter_v2 import select_candidates
except Exception:
    predict_next_poi = None
    select_candidates = None


def _load_similarity_index(dry_run: bool) -> dict:
    if SIMILARITY_CACHE.exists():
        with open(SIMILARITY_CACHE, "rb") as f:
            idx = pickle.load(f)
        logger.info(f"Loaded similarity index ({len(idx)} users).")
        return idx
    if dry_run:
        logger.info("No similarity cache; using empty index for dry-run.")
        return {}
    logger.error("No similarity cache found. Run build_profiles.py first.")
    sys.exit(1)


def _load_profiles(data_loader, dry_run: bool) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for path in PROFILES_CACHE_DIR.glob("*.json"):
        d = load_json(path)
        if d and "user_id" in d:
            profiles[d["user_id"]] = d
    if profiles:
        logger.info(f"Loaded {len(profiles)} cached profiles.")
        return profiles
    if dry_run:
        logger.info("No cached profiles; synthesising from raw stats for dry-run.")
        for uid in data_loader.get_all_user_ids():
            stats = data_loader.compute_user_stats(uid)
            visited = [c["loc_id"] for trip in data_loader.get_user_train_val_trips(uid) for c in trip]
            profiles[uid] = {
                "user_id": uid,
                "enhanced_profile": stats["stats_prompt"],
                "raw_stats_prompt": stats["stats_prompt"],
                "top_categories": stats["top_categories"],
                "visited_loc_ids": visited,
                "visited_geohashes": [],
                "num_trips": len(data_loader.get_user_train_val_trips(uid)),
            }
        return profiles
    logger.error("No cached profiles found. Run build_profiles.py first.")
    sys.exit(1)


def _pool_entry_for_save(c: dict) -> dict:
    """Strip destination_prior text (verbose) and keep analysis-relevant fields."""
    return {
        "loc_id": c["loc_id"],
        "name": c["name"],
        "category": c["category"],
        "dist_km": round(c["dist_km"], 4),
        "source": c["source"],
        "visit_count": c.get("visit_count", 0),
        "sim_weight": round(c.get("sim_weight", 0.0), 4),
        "heuristic_score": round(c.get("heuristic_score", 0.0), 4),
        "temporal_hint": c.get("temporal_hint", ""),
    }


def _save_pools(traj_id: str, raw_pool: list[dict], filtered_pool: list[dict], pool_dir: Path) -> None:
    """Persist raw and filtered pools for case study."""
    payload = {
        "traj_id": traj_id,
        "raw_pool_size": len(raw_pool),
        "filtered_pool_size": len(filtered_pool),
        "raw_pool": [_pool_entry_for_save(c) for c in raw_pool],
        "filtered_pool": [_pool_entry_for_save(c) for c in filtered_pool],
    }
    save_json(pool_dir / f"{traj_id}.json", payload)


def predict_all(
    data_loader,
    profiles: dict[str, dict],
    similarity_index: dict,
    traj_ids: list[str],
    dry_run: bool,
    workers: int,
    recompute: bool,
    transitions: dict,
    ablation: str = "full",
    pred_cache_dir: Path | None = None,
    pools_cache_dir: Path | None = None,
    quota_history: int | None = None,
    quota_nearby: int | None = None,
    use_social: bool = False,
) -> dict[str, list[int]]:
    pred_dir = pred_cache_dir or PREDICTIONS_V2_CACHE_DIR
    pool_dir = pools_cache_dir or POOLS_V2_CACHE_DIR
    pred_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)
    INTENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if select_candidates is None or predict_next_poi is None:
        logger.error("LLM prediction pipeline failed to import.")
        sys.exit(1)

    to_predict = []
    all_predictions: dict[str, list[int]] = {}

    for tid in traj_ids:
        cache_path = pred_dir / f"{tid}.json"
        if not recompute and cache_path.exists():
            d = load_json(cache_path)
            if d:
                all_predictions[tid] = d.get("ranked_loc_ids", [])
                continue
        to_predict.append(tid)

    logger.info(
        f"Prediction v2 ({ablation}): {len(all_predictions)} cached, {len(to_predict)} remaining (total {len(traj_ids)})"
    )

    traj_to_user: dict[str, str] = {}
    for uid, tids in data_loader.trips_by_user.items():
        for tid in tids:
            traj_to_user[tid] = uid

    def _predict_one(tid: str, metrics_out: list | None = None) -> tuple[str, list[int]]:
        context = data_loader.get_test_context(tid)
        if not context:
            return tid, []
        target_checkin = data_loader.get_test_target_checkin(tid)

        user_id = traj_to_user.get(tid)
        user_profile = profiles.get(user_id, {}) if user_id else {}

        if ablation in ("no_social", "no_priors"):
            similar_users = []
        else:
            similar_users = get_similar_users(user_id, similarity_index) if user_id else []

        raw_candidates = build_raw_pool(context, user_profile, similar_users, data_loader, ablation=ablation)

        candidates = select_candidates(
            tid,
            context,
            user_profile,
            similar_users,
            target_checkin,
            data_loader,
            transitions,
            dry_run=dry_run,
            ablation=ablation,
            metrics_out=metrics_out,
            quota_history=quota_history,
            quota_nearby=quota_nearby,
        )
        # Similar-user patterns are removed from the prediction LLM by default;
        # pass --use-social to feed them to the ranker (intent still uses them above).
        ranker_similar_users = similar_users if use_social else []
        result = predict_next_poi(
            tid,
            context,
            user_profile,
            ranker_similar_users,
            candidates,
            target_checkin,
            data_loader,
            dry_run=dry_run,
            ablation=ablation,
            metrics_out=metrics_out,
        )

        if dry_run:
            logger.info(f"\n{'='*60}\nPrompt for {tid}:\n{result.get('prompt', '')}\n{'='*60}")
            return tid, []

        # Save raw pool and filtered pool for case study / analysis
        _save_pools(tid, raw_candidates, candidates, pool_dir)

        if not result.get("ranked_loc_ids"):
            ranked = [c["loc_id"] for c in sorted(raw_candidates, key=lambda x: x.get("dist_km", 999.0))[:10]]
            result = {"traj_id": tid, "ranked_loc_ids": ranked, "mode": "distance_fallback"}

        save_json(pred_dir / f"{tid}.json", result)
        return tid, result.get("ranked_loc_ids", [])

    if not to_predict:
        return all_predictions

    progress = Progress(len(to_predict), "Predictions v2")
    if workers > 1 and not dry_run:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_predict_one, tid): tid for tid in to_predict}
            for future in concurrent.futures.as_completed(futures):
                tid, ranked = future.result()
                all_predictions[tid] = ranked
                progress.step()
    else:
        for tid in to_predict:
            tid, ranked = _predict_one(tid)
            all_predictions[tid] = ranked
            if not dry_run:
                progress.step()
            if dry_run:
                break

    return all_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Predict next POI (v2: quota-based candidate selection)"
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--traj-id", default=None, help="Predict for a single trajectory (debug)")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt without API call")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--recompute", action="store_true", help="Re-predict even if cached")
    parser.add_argument(
        "--ablation",
        default=None,
        help="Ablation variant or sensitivity tag (default: full). Variants write to isolated cache dirs.",
    )
    parser.add_argument(
        "--total-candidates",
        type=int,
        default=30,
        help="Total number of candidates (overrides QUOTA_HISTORY/QUOTA_NEARBY from config)",
    )
    parser.add_argument(
        "--history-ratio",
        type=float,
        default=0.9,
        help="Fraction of total candidates allocated to history (default: 0.9). Remainder goes to nearby.",
    )
    parser.add_argument(
        "--use-social",
        action="store_true",
        help="Feed similar-user patterns to the prediction LLM (default: off). "
        "Writes to an isolated *_social cache dir for ablation comparison.",
    )
    args = parser.parse_args()
    ablation = args.ablation or "full"

    # Compute quotas from CLI flags (None = use config defaults in select_candidates)
    quota_history: int | None = None
    quota_nearby: int | None = None
    total_tag: str = ""
    if args.total_candidates is not None:
        quota_history = int(args.total_candidates * args.history_ratio)
        quota_nearby = args.total_candidates - quota_history
        total_tag = f"_T{args.total_candidates}"
        logger.info(
            f"Candidate quotas: {quota_history} history + {quota_nearby} nearby "
            f"= {args.total_candidates} (ratio={args.history_ratio})"
        )

    # Derive cache dirs: when --ablation is set, use variant-specific dirs.
    # --use-social adds a _social suffix so the with-social run stays isolated.
    social_tag = "_social" if args.use_social else ""
    if args.ablation is None:
        if not total_tag and not social_tag:
            pred_cache_dir = PREDICTIONS_V2_CACHE_DIR
            pools_cache_dir = POOLS_V2_CACHE_DIR
        else:
            pred_cache_dir = CACHE_DIR / f"predictions_v2_full{total_tag}{social_tag}"
            pools_cache_dir = CACHE_DIR / f"pools_v2_full{total_tag}{social_tag}"
    else:
        pred_cache_dir = CACHE_DIR / f"predictions_v2_{ablation}{total_tag}{social_tag}"
        pools_cache_dir = CACHE_DIR / f"pools_v2_{ablation}{total_tag}{social_tag}"

    logger.info(f"Active dataset: {DATASET_TAG} ({DATA_DIR})")
    logger.info(f"Ablation     : {ablation}")
    logger.info(f"Use social   : {args.use_social}")
    logger.info(f"Predictions  → {pred_cache_dir}")
    logger.info(f"Pool snapshots → {pools_cache_dir}")

    data_loader = DataLoader()
    profiles = _load_profiles(data_loader, args.dry_run)
    similarity_index = _load_similarity_index(args.dry_run)

    transitions: dict = {}
    if TRANSITIONS_CACHE.exists():
        with open(TRANSITIONS_CACHE, "rb") as f:
            transitions = pickle.load(f)
        logger.info(f"Loaded transition table ({len(transitions)} source locations).")
    else:
        logger.warning("No transitions cache. Run build_profiles.py --build-transitions for best results.")

    traj_ids = [args.traj_id] if args.traj_id else data_loader.test_traj_ids

    predict_all(
        data_loader,
        profiles,
        similarity_index,
        traj_ids=traj_ids,
        dry_run=args.dry_run,
        workers=args.workers,
        recompute=args.recompute,
        transitions=transitions,
        ablation=ablation,
        pred_cache_dir=pred_cache_dir,
        pools_cache_dir=pools_cache_dir,
        quota_history=quota_history,
        quota_nearby=quota_nearby,
        use_social=args.use_social,
    )

    if not args.dry_run:
        logger.info(
            f"Prediction complete. Run: python evaluate.py --predictions-dir {pred_cache_dir}"
        )


if __name__ == "__main__":
    main()
