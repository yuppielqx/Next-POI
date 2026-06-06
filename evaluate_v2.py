"""
Stage 4: Evaluate predictions and compute Hit@K, N@K, and MRR metrics.

Usage:
  python evaluate.py [--predictions-dir PATH]   # default: cache/<dataset>/predictions/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATASET_TAG, DATA_DIR, EVALUATION_RESULTS, PREDICTIONS_CACHE_DIR, RESULTS_DIR
from src.data_loader import DataLoader
from src.evaluator import evaluate_and_save
from src.profile_builder import load_all_profiles
from src.utils import load_json, logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate next POI predictions")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name under datasets/ (e.g., nyc, tky, ca).",
    )
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help=f"Directory with prediction JSON files (default: {PREDICTIONS_CACHE_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test trajectories (in dataset order). Default: all.",
    )
    args = parser.parse_args()
    logger.info(f"Active dataset: {DATASET_TAG} ({DATA_DIR})")

    pred_dir = Path(args.predictions_dir) if args.predictions_dir else PREDICTIONS_CACHE_DIR

    # Load predictions
    all_predictions: dict[str, list[int]] = {}
    for path in pred_dir.glob("*.json"):
        d = load_json(path)
        if d and "traj_id" in d:
            all_predictions[d["traj_id"]] = d.get("ranked_loc_ids", [])

    if not all_predictions:
        logger.error(f"No prediction files found in {pred_dir}. Run predict.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(all_predictions)} predictions from {pred_dir}.")

    # Load data and profiles for stratum analysis
    data_loader = DataLoader()
    profiles = load_all_profiles()

    # Optionally restrict to the first N test trajectories (dataset order, reproducible)
    if args.limit is not None:
        ordered = [t for t in data_loader.test_traj_ids if t in all_predictions][: args.limit]
        all_predictions = {t: all_predictions[t] for t in ordered}
        logger.info(f"Limiting evaluation to first {len(all_predictions)} trajectories (--limit {args.limit}).")

    all_ground_truths = data_loader.get_test_ground_truths()
    ground_truths = {tid: gt for tid, gt in all_ground_truths.items() if tid in all_predictions}
    logger.info(f"Evaluating {len(ground_truths)} trajectories (matched with predictions).")
    # Derive output path: if custom predictions dir, write to results/<dataset>/evaluation_results_<suffix>.json
    limit_tag = f"_first{args.limit}" if args.limit is not None else ""
    if args.predictions_dir:
        suffix = Path(args.predictions_dir).name  # e.g. "predictions_v2_full_T40"
        out_path = RESULTS_DIR / f"evaluation_results_{suffix}{limit_tag}.json"
    else:
        out_path = EVALUATION_RESULTS if not limit_tag else RESULTS_DIR / f"evaluation_results{limit_tag}.json"
    results = evaluate_and_save(all_predictions, ground_truths, data_loader, profiles, output_path=out_path)

    # Print summary table
    overall = results.get("overall", {})
    print("\n=== Overall Metrics ===")
    for k, v in overall.items():
        print(f"  {k:20s}: {v}")

    by_stratum = results.get("by_stratum", {})
    for stratum, buckets in by_stratum.items():
        print(f"\n=== By {stratum} ===")
        for bucket, metrics in buckets.items():
            hit1 = metrics.get("Hit@1", 0)
            n10  = metrics.get("N@10", 0)
            mrr  = metrics.get("MRR", 0)
            n    = metrics.get("total", 0)
            print(f"  {bucket:12s}: Hit@1={hit1:.4f}  N@10={n10:.4f}  MRR={mrr:.4f}  (n={n})")


if __name__ == "__main__":
    main()
