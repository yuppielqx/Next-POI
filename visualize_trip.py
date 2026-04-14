"""
Visualize a single trajectory on a map with real tile background (contextily).

Usage:
  python visualize_trip.py --traj-id test_0
  python visualize_trip.py --traj-id test_0 --show-gt   # highlight ground truth
  python visualize_trip.py --traj-id test_0 --html       # interactive HTML (folium)
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader


def _checkins_to_rows(traj_id: str, data_loader: DataLoader) -> list[dict]:
    checkins = data_loader.trips[traj_id]
    rows = []
    for i, c in enumerate(checkins):
        rows.append({
            "idx": i + 1,
            "lat": c["lat"],
            "lon": c["lon"],
            "name": data_loader.get_poi_name(c["loc_id"]),
            "category": data_loader.get_poi_category(c["loc_id"]),
            "date": c["date"],
            "time": c["time"],
            "loc_id": c["loc_id"],
        })
    return rows


def _to_mercator(lon: float, lat: float) -> tuple[float, float]:
    """Convert WGS84 lon/lat to Web Mercator (EPSG:3857)."""
    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34 / 180
    return x, y


def _compute_offsets(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Alternate label positions to reduce overlap."""
    if not pts:
        return []
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_span = max(xs) - min(xs) or 1.0
    y_span = max(ys) - min(ys) or 1.0
    ox = x_span * 0.06
    oy = y_span * 0.06
    directions = [
        ( ox,  oy), (-ox,  oy), ( ox, -oy), (-ox, -oy),
        ( ox * 1.5, 0), (-ox * 1.5, 0),
    ]
    return [directions[i % len(directions)] for i in range(len(pts))]


# ── PNG with real tile background (contextily) ──────────────────────────────

def plot_png(traj_id: str, rows: list[dict], show_gt: bool, out_path: Path):
    import contextily as ctx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    xs, ys = zip(*[_to_mercator(r["lon"], r["lat"]) for r in rows])

    fig, ax = plt.subplots(figsize=(12, 10))

    # ── Path line ───────────────────────────────────────────────────────────
    ax.plot(xs, ys, color="#2563eb", linewidth=2.0, zorder=3,
            solid_capstyle="round", solid_joinstyle="round")

    # Direction arrows
    for i in range(len(rows) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1], ys[i + 1]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=1.5),
            zorder=4,
        )

    # ── Markers and labels ──────────────────────────────────────────────────
    offsets = _compute_offsets(list(zip(xs, ys)))

    for i, r in enumerate(rows):
        is_first = r["idx"] == 1
        is_last  = r["idx"] == len(rows)
        is_gt    = is_last and show_gt

        marker_color = "#16a34a" if is_first else ("#dc2626" if is_gt else "#2563eb")
        marker_size  = 120 if (is_first or is_gt) else 70

        ax.scatter(xs[i], ys[i], color=marker_color, s=marker_size,
                   zorder=5, edgecolors="white", linewidth=1.2)
        ax.text(xs[i], ys[i], str(r["idx"]),
                fontsize=6, color="white", ha="center", va="center",
                fontweight="bold", zorder=6)

        gt_tag = "  [GT]" if is_gt else ""
        label = (
            f"#{r['idx']} {r['name']}{gt_tag}\n"
            f"{r['category']}\n"
            f"{r['date']} {r['time']}"
        )
        ox, oy = offsets[i]
        box_color    = "#fef9c3" if is_gt else ("#dcfce7" if is_first else "white")
        border_color = "#ca8a04" if is_gt else ("#16a34a" if is_first else "#94a3b8")

        ax.annotate(
            label,
            xy=(xs[i], ys[i]),
            xytext=(xs[i] + ox, ys[i] + oy),
            fontsize=7.5,
            color="#1e293b",
            ha="left" if ox >= 0 else "right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=box_color,
                      edgecolor=border_color, linewidth=0.8, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color=border_color, lw=0.8),
            zorder=7,
        )

    # ── Basemap ─────────────────────────────────────────────────────────────
    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.Positron, zoom="auto")
    ax.set_axis_off()

    title = f"Trajectory: {traj_id}   ({len(rows)} check-ins)"
    ax.set_title(title, fontsize=13, pad=12, color="#1e293b", fontweight="semibold")

    legend_handles = [
        mpatches.Patch(color="#16a34a", label="Start"),
        mpatches.Patch(color="#2563eb", label="Check-in"),
    ]
    if show_gt:
        legend_handles.append(mpatches.Patch(color="#dc2626", label="Ground Truth"))
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right",
              framealpha=0.9, edgecolor="#cbd5e1")

    plt.tight_layout(pad=0.5)
    plt.savefig(str(out_path), dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved PNG → {out_path}")


# ── Multi-trip: PNG (user overview) ────────────────────────────────────────

def plot_user_png(user_id: str, trips_data: list[dict], out_path: Path):
    """
    trips_data: list of {traj_id, split, rows}
      split: "train" | "valid"
      rows: output of _checkins_to_rows()
    """
    import contextily as ctx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm

    n = len(trips_data)
    colors = [cm.tab20(i % 20) for i in range(n)]

    fig, ax = plt.subplots(figsize=(13, 11))

    for idx, trip in enumerate(trips_data):
        rows = trip["rows"]
        split = trip["split"]
        color = colors[idx]
        linestyle = "-" if split == "train" else "--"

        xs, ys = zip(*[_to_mercator(r["lon"], r["lat"]) for r in rows])

        ax.plot(xs, ys, color=color, linewidth=1.6, linestyle=linestyle,
                alpha=0.75, zorder=3)

        # Start marker (triangle up)
        ax.scatter(xs[0], ys[0], marker="^", color=color, s=60,
                   edgecolors="white", linewidth=0.8, zorder=5)
        # End marker (square)
        ax.scatter(xs[-1], ys[-1], marker="s", color=color, s=50,
                   edgecolors="white", linewidth=0.8, zorder=5)

        # Label first and last POI only
        for pos, r in [(0, rows[0]), (-1, rows[-1])]:
            label = f"{r['name']}\n{r['date']}"
            ox = (max(xs) - min(xs)) * 0.04 if pos == 0 else -(max(xs) - min(xs)) * 0.04
            oy = (max(ys) - min(ys)) * 0.04
            ax.annotate(
                label,
                xy=(xs[pos], ys[pos]),
                xytext=(xs[pos] + ox, ys[pos] + oy),
                fontsize=6,
                color="#1e293b",
                ha="left" if pos == 0 else "right",
                va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=color, linewidth=0.7, alpha=0.88),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
                zorder=6,
            )

    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.Positron, zoom="auto")
    ax.set_axis_off()

    n_train = sum(1 for t in trips_data if t["split"] == "train")
    n_valid = sum(1 for t in trips_data if t["split"] == "valid")
    ax.set_title(
        f"User {user_id} — {n} trajectories  "
        f"(train: {n_train}  valid: {n_valid})",
        fontsize=13, pad=12, color="#1e293b", fontweight="semibold",
    )

    legend_handles = [
        mpatches.Patch(color="#555", label="— train trajectory"),
        mpatches.Patch(color="#555", label="-- valid trajectory",
                       linestyle="--"),
        plt.scatter([], [], marker="^", color="grey", s=50, label="Trip start"),
        plt.scatter([], [], marker="s", color="grey", s=40, label="Trip end"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right",
              framealpha=0.9, edgecolor="#cbd5e1")

    plt.tight_layout(pad=0.5)
    plt.savefig(str(out_path), dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved user PNG → {out_path}")


# ── Multi-trip: HTML (user overview) ───────────────────────────────────────

def plot_user_html(user_id: str, trips_data: list[dict], out_path: Path):
    import folium

    all_lats = [r["lat"] for t in trips_data for r in t["rows"]]
    all_lons = [r["lon"] for t in trips_data for r in t["rows"]]
    center = (sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons))

    palette = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    ]

    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    for idx, trip in enumerate(trips_data):
        rows = trip["rows"]
        traj_id = trip["traj_id"]
        split = trip["split"]
        color = palette[idx % len(palette)]
        dash = "10 5" if split == "valid" else None

        folium.PolyLine(
            [(r["lat"], r["lon"]) for r in rows],
            color=color, weight=2.5, opacity=0.8,
            dash_array=dash,
            tooltip=f"{traj_id} ({split}, {len(rows)} checkins)",
        ).add_to(m)

        for endpoint_idx in (0, -1):
            r = rows[endpoint_idx]
            icon_name = "play" if endpoint_idx == 0 else "stop"
            folium.Marker(
                location=(r["lat"], r["lon"]),
                tooltip=f"{traj_id} {'start' if endpoint_idx == 0 else 'end'}: {r['name']}",
                popup=folium.Popup(
                    f"<b>{r['name']}</b><br>{r['category']}<br>{r['date']} {r['time']}",
                    max_width=200,
                ),
                icon=folium.Icon(color="gray", icon=icon_name, prefix="fa"),
            ).add_to(m)

    m.save(str(out_path))
    print(f"Saved user HTML → {out_path}")


# ── Interactive HTML (folium, single trip) ──────────────────────────────────

def plot_html(rows: list[dict], show_gt: bool, out_path: Path):
    import folium

    lats = [r["lat"] for r in rows]
    lons = [r["lon"] for r in rows]
    center = (sum(lats) / len(lats), sum(lons) / len(lons))

    from folium.plugins import PolyLineTextPath

    m = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron")
    coords = [(r["lat"], r["lon"]) for r in rows]
    line = folium.PolyLine(coords, color="#2563eb", weight=2.5, opacity=0.8)
    line.add_to(m)
    PolyLineTextPath(
        line, "   ➤", repeat=True, offset=0,
        attributes={"fill": "#2563eb", "font-weight": "bold", "font-size": "16"},
    ).add_to(m)

    for r in rows:
        is_first = r["idx"] == 1
        is_last  = r["idx"] == len(rows)
        is_gt    = is_last and show_gt
        color = "green" if is_first else ("red" if is_gt else "blue")
        icon  = "play"  if is_first else ("flag" if is_gt else "circle")
        popup_html = (
            f"<b>#{r['idx']} {r['name']}</b><br>"
            f"{r['category']}<br>"
            f"{r['date']} {r['time']}"
            + ("&nbsp;<b>[GT]</b>" if is_gt else "")
        )
        folium.Marker(
            location=(r["lat"], r["lon"]),
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"#{r['idx']} {r['name']}",
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        ).add_to(m)

    m.save(str(out_path))
    print(f"Saved HTML → {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize trajectories on a map")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--traj-id", help="Single trajectory ID (e.g. test_0)")
    group.add_argument("--user-id", help="User ID: visualize all train+valid trajectories")
    parser.add_argument("--show-gt", action="store_true",
                        help="Highlight the last check-in as ground truth (--traj-id only)")
    parser.add_argument("--html", action="store_true",
                        help="Output interactive HTML (folium) instead of PNG")
    parser.add_argument("--out", default=None, help="Output file path (auto-named if omitted)")
    args = parser.parse_args()

    data_loader = DataLoader()

    # ── Single trajectory ───────────────────────────────────────────────────
    if args.traj_id:
        if args.traj_id not in data_loader.trips:
            print(f"Error: traj_id '{args.traj_id}' not found.")
            sys.exit(1)

        rows = _checkins_to_rows(args.traj_id, data_loader)
        print(f"Trajectory {args.traj_id}: {len(rows)} check-ins")
        for r in rows:
            gt_tag = " [GT]" if (args.show_gt and r["idx"] == len(rows)) else ""
            print(f"  #{r['idx']} {r['date']} {r['time']}  {r['name']} ({r['category']}){gt_tag}")

        suffix = "html" if args.html else "png"
        out_path = Path(args.out) if args.out else Path(f"results/{args.traj_id}.{suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.html:
            plot_html(rows, args.show_gt, out_path)
        else:
            plot_png(args.traj_id, rows, args.show_gt, out_path)

    # ── All train+valid trajectories for a user ─────────────────────────────
    else:
        user_id = args.user_id
        traj_ids = [
            tid for tid in data_loader.trips_by_user.get(user_id, [])
            if tid.startswith("train_") or tid.startswith("valid_")
        ]
        if not traj_ids:
            print(f"Error: user_id '{user_id}' not found or has no train/valid trips.")
            sys.exit(1)

        # Merge all checkins into one chronological trip
        all_rows = []
        for tid in traj_ids:
            all_rows.extend(_checkins_to_rows(tid, data_loader))
        all_rows.sort(key=lambda r: (r["date"], r["time"]))
        for i, r in enumerate(all_rows):
            r["idx"] = i + 1

        print(f"User {user_id}: {len(traj_ids)} trajectories, {len(all_rows)} check-ins total")
        for r in all_rows:
            print(f"  #{r['idx']} {r['date']} {r['time']}  {r['name']} ({r['category']})")

        suffix = "html" if args.html else "png"
        out_path = Path(args.out) if args.out else Path(f"results/user_{user_id}.{suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.html:
            plot_html(all_rows, show_gt=False, out_path=out_path)
        else:
            plot_png(f"user_{user_id}", all_rows, show_gt=False, out_path=out_path)


if __name__ == "__main__":
    main()
