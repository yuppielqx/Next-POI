"""
Stage 1 + 2: Build user profiles and user similarity index.

Usage:
  python build_profiles.py [--user-id UID]        # single user (debug)
                           [--recompute]           # rebuild all from scratch
                           [--workers N]           # concurrent LLM profile builders
                           [--skip-similarity]     # only build profiles, skip similarity
                           [--migrate-temporal]    # add temporal_profile to existing cached profiles (no LLM)
                           [--build-transitions]   # build cache/transitions.pkl from all train+val trips
"""
import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import TOP_TRANSITIONS, TRANSITIONS_CACHE
from src.data_loader import DataLoader
from src.profile_builder import build_all_profiles, build_profile, load_all_profiles, migrate_temporal_profiles
from src.utils import logger


def build_transitions(data_loader) -> dict[int, list[tuple[int, int]]]:
    """
    Mine all train+val trips to build a global loc→next_loc transition table.
    Returns {loc_id: [(next_loc_id, count), ...]} sorted by count descending,
    capped at TOP_TRANSITIONS entries per source location.
    """
    counts: dict[int, Counter] = defaultdict(Counter)
    for uid in data_loader.get_all_user_ids():
        for trip in data_loader.get_user_train_val_trips(uid):
            for i in range(len(trip) - 1):
                src = trip[i]["loc_id"]
                dst = trip[i + 1]["loc_id"]
                counts[src][dst] += 1

    result = {
        src: counter.most_common(TOP_TRANSITIONS)
        for src, counter in counts.items()
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Build user profiles and similarity index")
    parser.add_argument("--user-id", type=str, default=None, help="Build profile for a single user (debug)")
    parser.add_argument("--recompute", action="store_true", help="Rebuild all caches from scratch")
    parser.add_argument("--workers", type=int, default=16, help="Parallel profile-building workers")
    parser.add_argument("--skip-similarity", action="store_true", help="Skip similarity computation")
    parser.add_argument("--migrate-temporal", action="store_true",
                        help="Add temporal_profile to existing cached profiles without re-calling LLM")
    parser.add_argument("--build-transitions", action="store_true",
                        help="Build cache/transitions.pkl from all train+val trips")
    args = parser.parse_args()

    data_loader = DataLoader()

    # ── Temporal migration (standalone, no LLM) ──────────────────────────────
    if args.migrate_temporal:
        n = migrate_temporal_profiles(data_loader)
        logger.info(f"Temporal migration complete: {n} profiles updated.")
        if not args.build_transitions:
            return

    # ── Transitions precomputation ────────────────────────────────────────────
    if args.build_transitions:
        TRANSITIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Building global transition table from all train+val trips…")
        transitions = build_transitions(data_loader)
        with open(TRANSITIONS_CACHE, "wb") as f:
            pickle.dump(transitions, f)
        logger.info(f"Transitions saved → {TRANSITIONS_CACHE} ({len(transitions)} source locations)")
        return

    # ── Stage 1: build user profiles ────────────────────────────
    if args.user_id:
        profile = build_profile(args.user_id, data_loader)
        logger.info(f"\nProfile for user {args.user_id}:\n{profile.get('enhanced_profile', '')}")
        return

    profiles = build_all_profiles(data_loader, force=args.recompute, workers=args.workers)
    logger.info(f"Stage 1 complete: {len(profiles)} user profiles ready.")

    if args.skip_similarity:
        return

    # ── Stage 2: build user similarity index ──────────────────────────────
    from src.user_similarity import build_similarity_index
    similarity_index = build_similarity_index(profiles, force=args.recompute)
    logger.info(f"Stage 2 complete: similarity index ready ({len(similarity_index)} users).")


if __name__ == "__main__":
    main()
