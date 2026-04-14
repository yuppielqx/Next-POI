"""
Stage 4: Evaluation.

Computes Hit@K, N@K, and MRR over all test predictions.
Also produces per-stratum breakdowns.
"""
import math
import json
from pathlib import Path

from src.config import EVAL_K_VALUES, EVALUATION_RESULTS
from src.utils import logger, save_json, time_of_day_label


def compute_metrics(
    predictions: dict[str, list[int]],
    ground_truths: dict[str, int],
) -> dict:
    """
    predictions: {traj_id: [loc_id_rank1, loc_id_rank2, …]} (up to 10)
    ground_truths: {traj_id: loc_id}
    Returns metrics dict.
    """
    hit = {k: 0 for k in EVAL_K_VALUES}
    ndcg = {k: 0.0 for k in EVAL_K_VALUES}
    mrr_total = 0.0
    n_in_candidates = 0  # how often GT was in the ranked list at all
    count = 0

    for traj_id, gt in ground_truths.items():
        pred = predictions.get(traj_id, [])
        if gt in pred:
            rank = pred.index(gt) + 1
            mrr_total += 1.0 / rank
            n_in_candidates += 1
            for k in EVAL_K_VALUES:
                if rank <= k:
                    hit[k] += 1
                    ndcg[k] += 1.0 / math.log2(rank + 1)
        count += 1

    if count == 0:
        return {}

    return {
        **{f"Hit@{k}": round(hit[k] / count, 4) for k in EVAL_K_VALUES},
        **{f"N@{k}": round(ndcg[k] / count, 4) for k in EVAL_K_VALUES},
        "MRR": round(mrr_total / count, 4),
        "recall_in_top10": round(n_in_candidates / count, 4),
        "total": count,
    }


def compute_stratum_metrics(
    predictions: dict[str, list[int]],
    ground_truths: dict[str, int],
    data_loader,
    profiles: dict[str, dict],
) -> dict:
    """Compute metrics broken down by user data richness, trip length, and time of day."""

    def user_id_for(traj_id: str) -> str | None:
        for uid, tids in data_loader.trips_by_user.items():
            if traj_id in tids:
                return uid
        return None

    strata: dict[str, dict[str, list]] = {
        "user_richness": {"sparse": [], "medium": [], "rich": []},
        "trip_length": {"short": [], "medium": [], "long": []},
        "time_of_day": {"morning": [], "afternoon": [], "evening": [], "night": []},
    }

    for traj_id, gt in ground_truths.items():
        pred = predictions.get(traj_id, [])
        context = data_loader.get_test_context(traj_id)

        # User richness
        uid = user_id_for(traj_id)
        if uid:
            p = profiles.get(uid, {})
            n_trips = p.get("num_trips", 0)
            bucket = "sparse" if n_trips <= 3 else ("medium" if n_trips <= 10 else "rich")
            strata["user_richness"][bucket].append((traj_id, gt, pred))

        # Trip length
        tlen = len(context) + 1  # full trip length
        tl_bucket = "short" if tlen <= 3 else ("medium" if tlen <= 6 else "long")
        strata["trip_length"][tl_bucket].append((traj_id, gt, pred))

        # Time of day of the target (last checkin)
        full_checkins = data_loader.trips[traj_id]
        if full_checkins:
            hour = full_checkins[-1]["hour"]
            tod = time_of_day_label(hour)
            strata["time_of_day"][tod].append((traj_id, gt, pred))

    results = {}
    for stratum_name, buckets in strata.items():
        results[stratum_name] = {}
        for bucket_name, items in buckets.items():
            if not items:
                continue
            bucket_preds = {tid: p for tid, _, p in items}
            bucket_gt = {tid: g for tid, g, _ in items}
            results[stratum_name][bucket_name] = compute_metrics(bucket_preds, bucket_gt)

    return results


def evaluate_and_save(
    predictions: dict[str, list[int]],
    ground_truths: dict[str, int],
    data_loader=None,
    profiles: dict[str, dict] | None = None,
) -> dict:
    """Run full evaluation and save to EVALUATION_RESULTS."""
    overall = compute_metrics(predictions, ground_truths)
    logger.info("=== Evaluation Results ===")
    for k, v in overall.items():
        logger.info(f"  {k}: {v}")

    result = {"overall": overall}

    if data_loader and profiles:
        strata = compute_stratum_metrics(predictions, ground_truths, data_loader, profiles)
        result["by_stratum"] = strata

    EVALUATION_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    save_json(EVALUATION_RESULTS, result)
    logger.info(f"Results saved to {EVALUATION_RESULTS}")
    return result
