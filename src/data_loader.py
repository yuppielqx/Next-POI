"""
Data loading and access layer for dataset directories in the project format.

All expensive loading happens once at DataLoader.__init__. After that,
all lookups are in-memory O(1) or O(log N) via KD-Tree.
"""
import ast
import csv
import json
import pickle
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from src.config import (
    DATA_DIR,
    LOC2ID_PATH,
    PROMPTS_REFINED_DIR,
    TRIPS_TEST,
    TRIPS_TRAIN,
    TRIPS_VALID,
    USER_INDEX_PATH,
)
from src.utils import haversine, logger, parse_timestamp


class DataLoader:
    def __init__(self):
        logger.info("Loading data…")
        self._load_loc2id()
        self._build_kdtree()
        self._load_trips()
        self._load_user_index()
        self._loc_metadata: dict[int, dict] = {}  # lazy cache for POI metadata
        logger.info(
            f"Ready: {len(self.loc2id)} POIs, "
            f"{sum(len(v) for v in self.trips.values())} trips, "
            f"{len(self.user_index)} users"
        )

    # ── loc2id / id2loc ────────────────────────────────────────────────────

    def _load_loc2id(self):
        with open(LOC2ID_PATH, "rb") as f:
            self.loc2id: dict[tuple[float, float], int] = pickle.load(f)
        # Build reverse mapping and coordinate arrays
        self.id2loc: dict[int, tuple[float, float]] = {v: k for k, v in self.loc2id.items()}
        self.all_loc_ids: list[int] = sorted(self.id2loc.keys())

    def resolve_checkin(self, checkin_str: str) -> dict:
        """
        Parse 'lon,lat,timestamp' string into a dict with resolved loc_id.
        Returns: {lon, lat, timestamp, loc_id, hour}
        """
        # split on first 2 commas only (timestamp may contain no commas but be safe)
        parts = checkin_str.strip().split(",", 2)
        lon, lat = float(parts[0]), float(parts[1])
        ts = parts[2].strip()
        loc_id = self.loc2id.get((lon, lat))
        if loc_id is None:
            # fallback: find nearest key
            loc_id = self._nearest_loc_id(lon, lat)
        dt = parse_timestamp(ts)
        return {
            "lon": lon,
            "lat": lat,
            "timestamp": ts,
            "loc_id": loc_id,
            "hour": dt.hour,
            "weekday": dt.weekday(),
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M"),
        }

    def _nearest_loc_id(self, lon: float, lat: float) -> int:
        """Fallback: find the loc_id whose coordinates are closest."""
        dists, idx = self._kdtree.query([lat, lon], k=1)
        return self.all_loc_ids[idx]

    # ── KD-Tree ────────────────────────────────────────────────────────────

    def _build_kdtree(self):
        """Build a KD-Tree over (lat, lon) for all POIs."""
        coords = np.array(
            [(self.id2loc[lid][1], self.id2loc[lid][0]) for lid in self.all_loc_ids]
        )  # shape (N, 2): [lat, lon]
        self._kdtree = KDTree(coords)
        self._kdtree_ids = self.all_loc_ids  # index → loc_id

    def get_nearby_pois(
        self, lon: float, lat: float, top_n: int = 100
    ) -> list[tuple[int, float]]:
        """
        Return top_n nearest POIs as [(loc_id, dist_km), …] sorted by distance.
        Uses KD-Tree for O(log N) query; distance computed via haversine.
        """
        k = min(top_n, len(self._kdtree_ids))
        dists, idxs = self._kdtree.query([lat, lon], k=k)
        if k == 1:
            dists, idxs = [dists], [idxs]
        result = []
        for idx, _approx_dist in zip(idxs, dists):
            lid = self._kdtree_ids[idx]
            plat, plon = self._kdtree.data[idx]
            dist_km = haversine(lon, lat, plon, plat)
            result.append((lid, dist_km))
        return result  # already sorted by KD-Tree distance

    # ── Trips ──────────────────────────────────────────────────────────────

    def _load_trips(self):
        """Load all splits into self.trips[traj_id] = list[dict] of checkins."""
        self.trips: dict[str, list[dict]] = {}
        self.trips_by_user: dict[str, list[str]] = {}  # user_id → [traj_id, …]

        for path, split in [
            (TRIPS_TRAIN, "train"),
            (TRIPS_VALID, "valid"),
            (TRIPS_TEST, "test"),
        ]:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    traj_id = row["traj_id"]
                    user_id = str(row["user_id"])
                    raw_trips = ast.literal_eval(row["trips"])
                    checkins = [self.resolve_checkin(c) for c in raw_trips]
                    self.trips[traj_id] = checkins
                    self.trips_by_user.setdefault(user_id, []).append(traj_id)

        # Separate test traj IDs
        self.test_traj_ids = [tid for tid in self.trips if tid.startswith("test_")]
        self.train_traj_ids = [tid for tid in self.trips if tid.startswith("train_")]
        self.valid_traj_ids = [tid for tid in self.trips if tid.startswith("valid_")]

    def get_user_train_val_trips(self, user_id: str) -> list[list[dict]]:
        """Return all train+val checkin sequences for a user."""
        result = []
        for tid in self.trips_by_user.get(user_id, []):
            if tid.startswith("train_") or tid.startswith("valid_"):
                result.append(self.trips[tid])
        return result

    # ── User stats (computed from trip data) ───────────────────────────────

    def compute_user_stats(self, user_id: str) -> dict:
        """
        Compute statistics for a user directly from their train+val trips.
        Returns:
          - stats_prompt: formatted text summary
          - top_categories: list of category strings by visit frequency
        """
        trips = self.get_user_train_val_trips(user_id)
        n_trips = len(trips)

        hour_counter: Counter = Counter()
        loc_counter: Counter = Counter()
        cat_counter: Counter = Counter()

        for trip in trips:
            for c in trip:
                hour_counter[c["hour"]] += 1
                loc_counter[c["loc_id"]] += 1
                cat = self.get_poi_category(c["loc_id"])
                if cat and cat.lower() != "unknown":
                    cat_counter[cat] += 1

        top_hours = "; ".join(
            f"{h:02d}:00-{h:02d}:59 for {cnt} times"
            for h, cnt in hour_counter.most_common(5)
        ) or "N/A"

        top_locs_parts = []
        for lid, cnt in loc_counter.most_common(5):
            name = self.get_poi_name(lid)
            lon, lat = self.id2loc[lid]
            top_locs_parts.append(f"'{name}' at ({lon}, {lat}) for {cnt} times")
        top_locs = "; ".join(top_locs_parts) or "N/A"

        top_cats_parts = [
            f"'{cat}' for {cnt} times"
            for cat, cnt in cat_counter.most_common(5)
        ]
        top_cats = "; ".join(top_cats_parts) or "N/A"

        stats_prompt = (
            f"The user with ID {user_id} has taken {n_trips} trips in total.\n"
            f"1. The top hours and frequencies for this user are: {top_hours}.\n"
            f"2. The top locations and frequencies are: {top_locs}.\n"
            f"3. The top categories and frequencies are: {top_cats}."
        )

        top_categories = [cat for cat, _ in cat_counter.most_common(10)]
        return {"stats_prompt": stats_prompt, "top_categories": top_categories}

    # ── User index ─────────────────────────────────────────────────────────

    def _load_user_index(self):
        """Load user_index.json: maps user_id → [[row_idx, traj_id], …]."""
        with open(USER_INDEX_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.user_index: dict[str, list] = {str(k): v for k, v in raw.items()}

    def get_all_user_ids(self) -> list[str]:
        return list(self.user_index.keys())

    # ── POI descriptions ───────────────────────────────────────────────────

    @lru_cache(maxsize=6000)
    def get_poi_description(self, loc_id: int) -> str:
        """Load full prompts_refined text for a location. Cached in memory."""
        path = PROMPTS_REFINED_DIR / f"{loc_id}.txt"
        if not path.exists():
            return f"[No description for location {loc_id}]"
        return path.read_text(encoding="utf-8").strip()

    def get_poi_snippet(self, loc_id: int, sentences: int = 2) -> str:
        """Return first N sentences of the POI description for use in prompts."""
        text = self.get_poi_description(loc_id)
        # Strip markdown header line
        lines = [l for l in text.split("\n") if l.strip() and not l.startswith("**")]
        combined = " ".join(lines[:6])
        # Split by sentence boundary
        parts = re.split(r"(?<=[.!?])\s+", combined)
        return " ".join(parts[:sentences])

    def get_poi_metadata(self, loc_id: int) -> dict:
        """Return {name, category, parent_category} by parsing first few lines."""
        if loc_id in self._loc_metadata:
            return self._loc_metadata[loc_id]
        text = self.get_poi_description(loc_id)
        # Extract name from header: **Location Analysis: NAME**
        name_match = re.search(r"\*\*Location Analysis:\s*(.+?)\*\*", text)
        name = name_match.group(1).strip() if name_match else f"Location {loc_id}"
        # Extract category from overview: "It is situated in the X category, specifically a Y."
        # or "specifically a Y" or "is a Y located at"
        specific_match = re.search(r"specifically an? ([^\.]+)", text)
        if specific_match:
            category = specific_match.group(1).strip()
        else:
            # "is a [adj] X [Y] located at" — capture 1-3 words before "located"
            cat_match = re.search(
                r"\bname\b.*?\bis an? (?:\w+ ){0,2}?(\w[\w &]+?) located", text, re.DOTALL
            )
            if not cat_match:
                # fallback: any "is a/an WORD+ located"
                cat_match = re.search(r"\bis an? ([\w][\w ,&]+?) located", text)
            # strip leading adjectives (popular, well-known, etc.)
            if cat_match:
                raw = cat_match.group(1).strip()
                # Take last 1-3 words as category
                words = raw.split()
                category = " ".join(words[-min(3, len(words)):])
            else:
                category = "Unknown"
        parent_match = re.search(r"situated in the ([^,\n]+?) category", text)
        parent = parent_match.group(1).strip() if parent_match else ""
        meta = {"name": name, "category": category, "parent_category": parent}
        self._loc_metadata[loc_id] = meta
        return meta

    def get_poi_name(self, loc_id: int) -> str:
        return self.get_poi_metadata(loc_id)["name"]

    def get_poi_category(self, loc_id: int) -> str:
        return self.get_poi_metadata(loc_id)["category"]

    # ── Ground truth for evaluation ────────────────────────────────────────

    def get_test_ground_truths(self) -> dict[str, int]:
        """
        For each test trajectory, return {traj_id: ground_truth_loc_id}.
        Ground truth = last checkin's loc_id.
        Trajectories with < 2 checkins are excluded.
        """
        gt = {}
        for tid in self.test_traj_ids:
            checkins = self.trips[tid]
            if len(checkins) >= 2:
                gt[tid] = checkins[-1]["loc_id"]
        return gt

    def get_test_context(self, traj_id: str) -> list[dict]:
        """Return all but the last checkin (prediction context) for a test trip."""
        checkins = self.trips[traj_id]
        return checkins[:-1]

    def get_test_target_checkin(self, traj_id: str) -> dict | None:
        """Return the held-out final checkin for a test trip, including its timestamp."""
        checkins = self.trips.get(traj_id, [])
        if len(checkins) < 2:
            return None
        return checkins[-1]
