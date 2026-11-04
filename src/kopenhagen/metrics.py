"""Per-agent and per-run measures of how close people come to the glass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from kopenhagen.geometry import Layout, glass

HEAD_ZONE = (-1.5, 1.5)  # y range around the nosing where the turn happens
BRUSH = 0.10  # clearance below this counts as brushing the glass (m)


def clearance_to_glass(lay: Layout, traj: pd.DataFrame, radius: float) -> pd.DataFrame:
    """Add body-edge clearance to the left and right glass to each row."""
    pts = shapely.points(traj["x"].to_numpy(), traj["y"].to_numpy())
    out = traj.copy()
    for side, line in glass(lay).items():
        out[f"clr_{side}"] = shapely.distance(pts, line) - radius
    out["clr_glass"] = out[["clr_left", "clr_right"]].min(axis=1)
    return out


def per_agent(lay: Layout, traj: pd.DataFrame, radius: float) -> pd.DataFrame:
    """Minimum clearance to the glass inside the stair-head zone, per agent."""
    df = clearance_to_glass(lay, traj, radius)
    head = df[(df["y"] > HEAD_ZONE[0]) & (df["y"] < HEAD_ZONE[1])]
    agg = head.groupby("id").agg(
        min_clr=("clr_glass", "min"),
        min_clr_left=("clr_left", "min"),
        min_clr_right=("clr_right", "min"),
    )
    times = df.groupby("id")["t"].agg(["min", "max"])
    agg["travel_time"] = (times["max"] - times["min"]).reindex(agg.index)
    agg = agg.join(step_entry(df))
    return agg.reset_index()


def step_entry(traj: pd.DataFrame) -> pd.DataFrame:
    """State of each person at the first sample past the top step (y < 0).

    entry_angle: angle between walking direction and straight down the stair (deg).
    entry_speed: speed (m/s); entry_x: lateral position (m).
    """
    below = traj[traj["y"] < 0].sort_values("t").groupby("id").first()
    angle = np.degrees(np.abs(np.arctan2(below["vx"], -below["vy"])))
    return pd.DataFrame({
        "entry_angle": angle,
        "entry_speed": np.hypot(below["vx"], below["vy"]),
        "entry_x": below["x"],
    })


def summary(agents: pd.DataFrame, traj: pd.DataFrame) -> dict[str, float]:
    return {
        "median_min_clr": float(agents["min_clr"].median()),
        "share_brushing": float((agents["min_clr"] < BRUSH).mean()),
        "median_travel_time": float(agents["travel_time"].median()),
        "median_entry_angle": float(agents["entry_angle"].median()),
        "median_entry_speed": float(agents["entry_speed"].median()),
        "total_time": float(traj["t"].max()),
        "stuck": int(traj.attrs.get("stuck", 0)),
        "failed": bool(traj.attrs.get("failed", False)),
    }


def heading_change_rate(traj: pd.DataFrame) -> np.ndarray:
    """Absolute turning rate (rad/s) per recorded sample, for curvature plots."""
    rates = []
    for _, g in traj.sort_values("t").groupby("id"):
        ang = np.unwrap(np.arctan2(g["vy"].to_numpy(), g["vx"].to_numpy()))
        dt = np.diff(g["t"].to_numpy())
        rates.append(np.abs(np.diff(ang)) / np.where(dt > 0, dt, np.nan))
    return np.concatenate(rates) if rates else np.array([])


DENSITY_BOX = (-0.5, 2.0)  # y range of the stair-head area used for density


def _crossings(traj: pd.DataFrame, group: str, below: bool) -> np.ndarray:
    g = traj[traj["group"] == group]
    side = g[g["y"] < 0] if below else g[g["y"] > 0]
    return np.sort(side.groupby("id")["t"].min().to_numpy())


def _flow(times: np.ndarray) -> float:
    """Persons per second over the central 80 % of crossings."""
    if len(times) < 5:
        return float("nan")
    lo, hi = int(0.1 * len(times)), int(0.9 * len(times))
    span = times[hi] - times[lo]
    return float((hi - lo) / span) if span > 0 else float("nan")


def stair_head_density(lay: Layout, traj: pd.DataFrame) -> pd.Series:
    """Persons per m² on the stair head, per recorded time."""
    from shapely import box

    from kopenhagen.geometry import walkable_area

    y0, y1 = DENSITY_BOX
    area = walkable_area(lay).intersection(box(lay.x_left, y0, lay.x_right, y1)).area
    inside = traj["x"].between(lay.x_left, lay.x_right) & traj["y"].between(y0, y1)
    counts = traj[inside].groupby("t").size().reindex(traj["t"].unique(), fill_value=0)
    return counts / area


def crowd_summary(lay: Layout, traj: pd.DataFrame, radius: float) -> dict[str, float]:
    agents = per_agent(lay, traj[traj["group"] == "main"], radius)
    dens = stair_head_density(lay, traj)
    busy = dens[dens > 0]
    times = traj.groupby(["group", "id"])["t"].agg(["min", "max"])
    travel = (times["max"] - times["min"]).groupby(level="group").median()
    return {
        "clear_time": float(traj["t"].max()),
        "flow_down": _flow(_crossings(traj, "main", below=True)),
        "flow_up": _flow(_crossings(traj, "up", below=False)),
        "density_mean": float(busy.mean()) if len(busy) else 0.0,
        "density_p95": float(busy.quantile(0.95)) if len(busy) else 0.0,
        "travel_down": float(travel.get("main", np.nan)),
        "travel_up": float(travel.get("up", np.nan)),
        "median_min_clr": float(agents["min_clr"].median()),
        "share_brushing": float((agents["min_clr"] < BRUSH).mean()),
        "stuck": int(traj.attrs.get("stuck", 0)),
        "failed": bool(traj.attrs.get("failed", False)),
    }
