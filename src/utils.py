import json
import math
import os
import tempfile
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nextpoi")


# ── Distance ───────────────────────────────────────────────────────────────

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return great-circle distance in km between two (lon, lat) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def movement_direction(checkins: list[dict]) -> str:
    """Summarise movement direction from last ≤3 checkins as a short phrase."""
    if len(checkins) < 2:
        return "stationary"
    recent = checkins[-3:] if len(checkins) >= 3 else checkins
    dlat = recent[-1]["lat"] - recent[0]["lat"]
    dlon = recent[-1]["lon"] - recent[0]["lon"]
    ns = "north" if dlat > 0 else "south"
    ew = "east" if dlon > 0 else "west"
    if abs(dlat) < 0.001 and abs(dlon) < 0.001:
        return "staying in the same area"
    if abs(dlat) < 0.001:
        return f"moving {ew}"
    if abs(dlon) < 0.001:
        return f"moving {ns}"
    return f"moving {ns}-{ew}"


# ── Time ───────────────────────────────────────────────────────────────────

def parse_timestamp(ts: str) -> datetime:
    """Parse '2012-04-08 16:02:10' to datetime."""
    return datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")


def hour_of_day(ts: str) -> int:
    return parse_timestamp(ts).hour


def time_of_day_label(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


# ── JSON cache (atomic write) ───────────────────────────────────────────────

def load_json(path: Path) -> Any | None:
    """Return parsed JSON or None if file does not exist."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """Atomically write JSON to path (temp-file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


# ── Rate limiter ───────────────────────────────────────────────────────────

class RateLimiter:
    """Simple token-bucket rate limiter (requests per minute)."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self._last = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last = time.monotonic()


# ── Progress logger ────────────────────────────────────────────────────────

class Progress:
    def __init__(self, total: int, label: str = ""):
        self.total = total
        self.label = label
        self.done = 0
        self._start = time.monotonic()

    def step(self, n: int = 1):
        self.done += n
        elapsed = time.monotonic() - self._start
        rate = self.done / elapsed if elapsed > 0 else 0
        eta = (self.total - self.done) / rate if rate > 0 else float("inf")
        eta_str = f"{eta:.0f}s" if eta < 3600 else f"{eta/3600:.1f}h"
        logger.info(
            f"{self.label} {self.done}/{self.total} "
            f"({100*self.done/self.total:.1f}%) ETA {eta_str}"
        )
