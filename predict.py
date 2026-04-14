"""
Stage 3: Predict the next POI for all test trajectories.

Usage:
  python predict.py [--traj-id TRAJID]
                   [--dry-run]
                   [--workers N]
                   [--mode {auto,llm,reranker,hybrid}]
                   [--recompute]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.candidate_selector import build_raw_pool
from src.config import (
    FORCED_INCLUDE_N,
    HYBRID_TOP_N,
    INTENT_CACHE_DIR,
    PREDICTIONS_CACHE_DIR,
    PREFILTER_CACHE_DIR,
    PROFILES_CACHE_DIR,
    SIMILARITY_CACHE,
    TRANSITIONS_CACHE,
)
from src.data_loader import DataLoader
from src.prior_bank import build_or_load_poi_embeddings
from src.user_similarity import get_similar_users
from src.utils import Progress, load_json, logger, save_json

try:
    from src.latent_reranker import build_prefix_embedding, load_or_build_artifacts, score_candidates_with_model
except Exception:
    build_prefix_embedding = None
    load_or_build_artifacts = None
    score_candidates_with_model = None

try:
    from src.llm_agent import predict_next_poi
    from src.llm_prefilter import select_candidates
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


def predict_all(
    data_loader,
    profiles: dict[str, dict],
    similarity_index: dict,
    traj_ids: list[str],
    dry_run: bool,
    workers: int,
    recompute: bool,
    transitions: dict,
    mode: str,
    hybrid_top_n: int,
    forced_include_n: int,
) -> dict[str, list[int]]:
    PREDICTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INTENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PREFILTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    reranker = None
    poi_artifacts = None
    prior_index = None
    if load_or_build_artifacts and build_prefix_embedding and score_candidates_with_model:
        reranker = load_or_build_artifacts(data_loader, force=False)
        if reranker:
            poi_artifacts = build_or_load_poi_embeddings(data_loader, force=False)
            prior_index = reranker.prior_index
    reranker_ready = bool(reranker and prior_index and poi_artifacts and build_prefix_embedding and score_candidates_with_model)

    if mode == "reranker" and not reranker_ready:
        logger.error("Prediction mode 'reranker' requires trained reranker artifacts. Run train_reranker.py first.")
        sys.exit(1)
    if mode == "hybrid" and not reranker_ready:
        logger.error("Prediction mode 'hybrid' requires trained reranker artifacts. Run train_reranker.py first.")
        sys.exit(1)
    if mode in {"llm", "hybrid"} and (select_candidates is None or predict_next_poi is None):
        logger.error(f"Prediction mode '{mode}' requires the LLM pipeline, but it failed to import.")
        sys.exit(1)

    to_predict = []
    all_predictions: dict[str, list[int]] = {}

    for tid in traj_ids:
        cache_path = PREDICTIONS_CACHE_DIR / f"{tid}.json"
        if not recompute and cache_path.exists():
            d = load_json(cache_path)
            if d:
                all_predictions[tid] = d.get("ranked_loc_ids", [])
                continue
        to_predict.append(tid)

    logger.info(
        f"Prediction: {len(all_predictions)} cached, {len(to_predict)} remaining (total {len(traj_ids)})"
    )

    traj_to_user: dict[str, str] = {}
    for uid, tids in data_loader.trips_by_user.items():
        for tid in tids:
            traj_to_user[tid] = uid

    def _predict_one(tid: str) -> tuple[str, list[int]]:
        context = data_loader.get_test_context(tid)
        if not context:
            return tid, []
        target_checkin = data_loader.get_test_target_checkin(tid)

        user_id = traj_to_user.get(tid)
        user_profile = profiles.get(user_id, {}) if user_id else {}
        similar_users = get_similar_users(user_id, similarity_index) if user_id else []
        raw_candidates = build_raw_pool(context, user_profile, similar_users, data_loader)

        path_mode = mode
        if path_mode == "auto":
            path_mode = "reranker" if reranker_ready else "llm"

        if path_mode == "reranker":
            prefix_emb = build_prefix_embedding(context, data_loader)
            scored = score_candidates_with_model(
                reranker.model,
                prefix_emb,
                raw_candidates,
                context,
                data_loader,
                poi_artifacts,
                prior_index,
                target_checkin=target_checkin,
                exclude_traj_id=tid,
                device=reranker.device,
            )

            if dry_run:
                logger.info(f"\n{'='*60}\nLatent reranker preview for {tid}:")
                for i, cand in enumerate(scored[:10], 1):
                    logger.info(
                        f"{i}. {cand['name']} | score={cand['reranker_score']:.4f} | support={cand.get('retrieval_support_count', 0.0):.0f}"
                    )
                return tid, []

            ranked = [c["loc_id"] for c in scored[:10]]
            result = {
                "traj_id": tid,
                "ranked_loc_ids": ranked,
                "mode": "latent_reranker",
                "top_candidates": scored[:10],
            }
            save_json(PREDICTIONS_CACHE_DIR / f"{tid}.json", result)
            return tid, ranked

        if path_mode == "hybrid":
            candidates = select_candidates(
                tid,
                context,
                user_profile,
                similar_users,
                target_checkin,
                data_loader,
                transitions,
                forced_include_n=forced_include_n,
                dry_run=dry_run,
            )
            prefix_emb = build_prefix_embedding(context, data_loader)
            reranked = score_candidates_with_model(
                reranker.model,
                prefix_emb,
                candidates,
                context,
                data_loader,
                poi_artifacts,
                prior_index,
                target_checkin=target_checkin,
                exclude_traj_id=tid,
                device=reranker.device,
            )
            hybrid_candidates = reranked[:hybrid_top_n]
            result = predict_next_poi(
                tid,
                context,
                user_profile,
                similar_users,
                hybrid_candidates,
                target_checkin,
                data_loader,
                dry_run=dry_run,
            )

            if dry_run:
                logger.info(f"\n{'='*60}\nPrompt for {tid}:\n{result.get('prompt', '')}\n{'='*60}")
                return tid, []

            result["mode"] = "hybrid"
            result["hybrid_candidates"] = [
                {
                    "loc_id": c["loc_id"],
                    "name": c["name"],
                    "reranker_score": c.get("reranker_score"),
                    "retrieval_support_count": c.get("retrieval_support_count"),
                }
                for c in hybrid_candidates
            ]
            save_json(PREDICTIONS_CACHE_DIR / f"{tid}.json", result)
            return tid, result.get("ranked_loc_ids", [])

        if select_candidates is None or predict_next_poi is None:
            ranked = [c["loc_id"] for c in sorted(raw_candidates, key=lambda x: x.get("dist_km", 999.0))[:10]]
            if dry_run:
                logger.info("%s\nFallback candidate ranking for %s: %s\n%s", '=' * 60, tid, ranked, '=' * 60)
                return tid, []
            result = {"traj_id": tid, "ranked_loc_ids": ranked, "mode": "distance_fallback"}
            save_json(PREDICTIONS_CACHE_DIR / f"{tid}.json", result)
            return tid, ranked

        candidates = select_candidates(
            tid,
            context,
            user_profile,
            similar_users,
            target_checkin,
            data_loader,
            transitions,
            forced_include_n=forced_include_n,
            dry_run=dry_run,
        )
        result = predict_next_poi(
            tid,
            context,
            user_profile,
            similar_users,
            candidates,
            target_checkin,
            data_loader,
            dry_run=dry_run,
        )

        if dry_run:
            logger.info(f"\n{'='*60}\nPrompt for {tid}:\n{result.get('prompt', '')}\n{'='*60}")
            return tid, []

        save_json(PREDICTIONS_CACHE_DIR / f"{tid}.json", result)
        return tid, result.get("ranked_loc_ids", [])

    if not to_predict:
        return all_predictions

    progress = Progress(len(to_predict), "Predictions")
    if workers > 1 and not dry_run and mode != "reranker":
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
    parser = argparse.ArgumentParser(description="Predict next POI for test trajectories")
    parser.add_argument("--traj-id", default=None, help="Predict for a single trajectory (debug)")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt without API call")
    parser.add_argument("--workers", type=int, default=32, help="Parallel prediction workers")
    parser.add_argument(
        "--mode",
        choices=["auto", "llm", "reranker", "hybrid"],
        default="auto",
        help="Prediction mode: auto prefers reranker if available, hybrid uses reranker shortlist + LLM final rank.",
    )
    parser.add_argument(
        "--hybrid-top-n",
        type=int,
        default=HYBRID_TOP_N,
        help="In hybrid mode, reranker shortlist size passed to the final LLM ranker.",
    )
    parser.add_argument(
        "--forced-include-n",
        type=int,
        default=FORCED_INCLUDE_N,
        help="How many nearest nearby candidates are forcibly kept in the LLM shortlist.",
    )
    parser.add_argument("--recompute", action="store_true", help="Re-predict even if cached")
    args = parser.parse_args()

    data_loader = DataLoader()
    profiles = _load_profiles(data_loader, args.dry_run)
    similarity_index = _load_similarity_index(args.dry_run)

    transitions: dict = {}
    if TRANSITIONS_CACHE.exists():
        with open(TRANSITIONS_CACHE, "rb") as f:
            transitions = pickle.load(f)
        logger.info(f"Loaded transition table ({len(transitions)} source locations).")
    else:
        logger.warning("No transitions cache found; run build_profiles.py --build-transitions for best results.")

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
        mode=args.mode,
        hybrid_top_n=args.hybrid_top_n,
        forced_include_n=args.forced_include_n,
    )

    if not args.dry_run:
        logger.info("Prediction complete. Run evaluate.py to compute metrics.")


if __name__ == "__main__":
    main()
