"""Run all simulations and write tables to results/.

Usage: .venv/bin/python scripts/run_study.py [label ...]
Labels: scenarios, sweep, direction, speed, crowd, counterflow (default: all).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from kopenhagen.geometry import SCENARIOS, Layout
from kopenhagen.metrics import crowd_summary, per_agent, summary
from kopenhagen.simulate import RunConfig, run

OUT = Path(__file__).resolve().parent.parent / "results"
SEEDS = range(8)
N_AGENTS = 40
MODELS = ("sfm", "warp", "cfsm")
SPEEDS = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)  # walking speed on the floor (m/s)


def one(lay: Layout, cfg: RunConfig, tags: dict) -> tuple[dict, pd.DataFrame]:
    traj = run(lay, cfg)
    agents = per_agent(lay, traj, cfg.radius)
    for k, v in tags.items():
        agents[k] = v
    row = {**tags, **summary(agents, traj)}
    return row, agents


def collect(jobs) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, agents = [], []
    for lay, cfg, tags in jobs:
        row, ag = one(lay, cfg, tags)
        rows.append(row)
        agents.append(ag)
    return pd.DataFrame(rows), pd.concat(agents, ignore_index=True)


def scenario_jobs():
    for name, lay in SCENARIOS.items():
        for model in ("sfm", "cfsm"):
            for inflow in (0.3, 0.8, 1.5):
                for seed in SEEDS:
                    cfg = RunConfig(model=model, inflow=inflow, n_agents=N_AGENTS, seed=seed)
                    yield lay, cfg, {"scenario": name, "model": model, "inflow": inflow, "seed": seed}


def sweep_jobs():
    base = Layout()
    sweeps = {
        "column_setback": [0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5],
        "column_offset": [-0.6, -0.3, 0.0, 0.1, 0.3, 0.5, 0.75],
        "stair_width": [1.6, 1.8, 2.0, 2.2, 2.4],
        "column_diameter": [0.3, 0.4, 0.5, 0.6, 0.7],
        "glass_end": [2.0, 2.3, 2.6, 3.0, 3.5],
    }
    for param, values in sweeps.items():
        for value in values:
            for has_column in (True, False):
                lay = base.with_(**{param: value, "has_column": has_column})
                for seed in SEEDS:
                    cfg = RunConfig(model="sfm", inflow=0.8, n_agents=N_AGENTS, seed=seed)
                    tags = {"param": param, "value": value, "has_column": has_column, "seed": seed}
                    yield lay, cfg, tags


def speed_jobs():
    for model in MODELS:
        names = list(SCENARIOS) if model == "sfm" else ["as built", "no column"]
        for name in names:
            for speed in SPEEDS:
                for seed in SEEDS:
                    cfg = RunConfig(model=model, floor_speed=speed, inflow=0.15, n_agents=20, seed=seed)
                    tags = {"model": model, "scenario": name, "speed": speed, "seed": seed}
                    yield SCENARIOS[name], cfg, tags


def direction_jobs():
    for direction in ("right", "down", "gallery", "up"):
        for name in ("as built", "no column"):
            for seed in SEEDS:
                cfg = RunConfig(direction=direction, inflow=0.8, n_agents=N_AGENTS, seed=seed)
                tags = {"direction": direction, "scenario": name, "seed": seed}
                yield SCENARIOS[name], cfg, tags


def example_trajectories() -> pd.DataFrame:
    frames = []
    for name, lay in SCENARIOS.items():
        for speed in (1.2, 1.8):
            traj = run(lay, RunConfig(floor_speed=speed, inflow=0.15, n_agents=20, seed=0))
            traj["scenario"], traj["speed"] = name, speed
            frames.append(traj)
    return pd.concat(frames, ignore_index=True)


def model_trajectories() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        for name in ("as built", "no column"):
            cfg = RunConfig(model=model, floor_speed=1.8, inflow=0.15, n_agents=20, seed=0)
            traj = run(SCENARIOS[name], cfg)
            traj["model"], traj["scenario"] = model, name
            frames.append(traj)
    return pd.concat(frames, ignore_index=True)


CROWD_SEEDS = range(5)
CROWD = {"n_agents": 120, "wide_spawn": True, "bypass_radius": 0.8, "max_time": 400.0}


def crowd_jobs():
    """Plane unloads: 120 people arrive in a burst and walk down."""
    for model in MODELS:
        for name in ("as built", "no column"):
            for inflow in (1.0, 1.5, 2.0):
                for seed in CROWD_SEEDS:
                    cfg = RunConfig(model=model, inflow=inflow, seed=seed, **CROWD)
                    yield SCENARIOS[name], cfg, {"model": model, "scenario": name, "inflow": inflow, "seed": seed}


def counterflow_jobs():
    """A few people come up the stair while others go down (10 people: no jams without the column
    for the social force model and WarpDriver)."""
    small = {**CROWD, "n_agents": 10, "max_time": 200.0}
    for model in MODELS:
        for name in ("as built", "no column"):
            for up_share in (0.25, 0.5):
                for seed in range(10):
                    cfg = RunConfig(model=model, inflow=0.8, up_share=up_share, seed=seed, **small)
                    tags = {"model": model, "scenario": name, "up_share": up_share, "seed": seed}
                    yield SCENARIOS[name], cfg, tags


def collect_crowd(jobs) -> pd.DataFrame:
    rows = []
    for lay, cfg, tags in jobs:
        traj = run(lay, cfg)
        rows.append({**tags, **crowd_summary(lay, traj, cfg.radius)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    for label, jobs in (
        ("scenarios", scenario_jobs()),
        ("sweep", sweep_jobs()),
        ("direction", direction_jobs()),
        ("speed", speed_jobs()),
    ):
        if len(sys.argv) > 1 and label not in sys.argv[1:]:
            continue
        runs, agents = collect(jobs)
        runs.to_csv(OUT / f"{label}_runs.csv", index=False)
        agents.to_csv(OUT / f"{label}_agents.csv", index=False)
        print(f"{label}: {len(runs)} runs, {runs['stuck'].sum()} stuck agents, {runs['failed'].sum()} failed runs")
    for label, jobs in (("crowd", crowd_jobs()), ("counterflow", counterflow_jobs())):
        if len(sys.argv) > 1 and label not in sys.argv[1:]:
            continue
        runs = collect_crowd(jobs)
        runs.to_csv(OUT / f"{label}_runs.csv", index=False)
        print(f"{label}: {len(runs)} runs, {runs['stuck'].sum()} stuck agents, {runs['failed'].sum()} failed runs")
    if len(sys.argv) == 1 or "speed" in sys.argv:
        model_trajectories().to_csv(OUT / "model_trajectories.csv", index=False)
    if len(sys.argv) == 1:
        example_trajectories().to_csv(OUT / "example_trajectories.csv", index=False)


if __name__ == "__main__":
    main()
