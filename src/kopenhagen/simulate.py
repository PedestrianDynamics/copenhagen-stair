"""Run one JuPedSim simulation of people walking through the stair head."""

from __future__ import annotations

from dataclasses import dataclass

import jupedsim as jps
import numpy as np
import pandas as pd

from kopenhagen.geometry import (
    Layout,
    column_bypass,
    exit_zone,
    far_side_waypoint,
    lane_waypoints,
    spawn_zone,
    walkable_area,
)

DT = 0.01
RECORD_EVERY = 10  # 0.1 s


@dataclass(frozen=True)
class RunConfig:
    model: str = "sfm"  # "sfm" (second order, inertia), "cfsm" (first order), "warp" (WarpDriver)
    direction: str = "right"  # "right" (observed path), "down" (from behind), "gallery", "up"
    inflow: float = 0.5  # persons per second entering
    n_agents: int = 40
    floor_speed: float = 1.3
    stair_speed: float = 0.7
    radius: float = 0.25
    spawn_centred: bool = False  # approach only along the stair axis
    wide_spawn: bool = False  # larger arrival area on the right, for crowds
    up_share: float = 0.0  # share of arrivals coming up the stair and leaving to the right
    bypass_radius: float = 0.4  # how close to the bypass waypoint counts as passed
    keep_right: bool = False  # stair lanes: down and up each keep to their right
    luggage_share: float = 0.0  # share of people carrying luggage: wider and slower
    luggage_radius: float = 0.30
    luggage_speed_factor: float = 0.8
    seed: int = 0
    max_time: float = 300.0


def _make_model(cfg: RunConfig):
    if cfg.model == "sfm":
        return jps.SocialForceModel()
    if cfg.model == "cfsm":
        return jps.CollisionFreeSpeedModel()
    if cfg.model == "warp":
        return jps.WarpDriverModel()
    raise ValueError(f"unknown model {cfg.model!r}")


def _agent_params(cfg: RunConfig, pos, journey_id, stage_id, heading, radius=None):
    radius = cfg.radius if radius is None else radius
    if cfg.model == "sfm":
        return jps.SocialForceModelAgentParameters(
            position=pos,
            orientation=heading,
            journey_id=journey_id,
            stage_id=stage_id,
            desired_speed=cfg.floor_speed,
            reaction_time=0.5,
            radius=radius,
        )
    if cfg.model == "warp":
        return jps.WarpDriverModelAgentParameters(
            position=pos,
            orientation=heading,
            journey_id=journey_id,
            stage_id=stage_id,
            desired_speed=cfg.floor_speed,
            radius=radius,
        )
    return jps.CollisionFreeSpeedModelAgentParameters(
        position=pos,
        journey_id=journey_id,
        stage_id=stage_id,
        desired_speed=cfg.floor_speed,
        radius=radius,
    )


def _arrival_times(cfg: RunConfig, rng: np.random.Generator) -> np.ndarray:
    gaps = rng.exponential(1.0 / cfg.inflow, size=cfg.n_agents)
    return np.cumsum(gaps) - gaps[0]


def _free_spot(zone, occupied, rng, min_dist):
    minx, miny, maxx, maxy = zone.bounds
    for _ in range(30):
        p = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > min_dist**2 for q in occupied):
            return p
    return None


def _set_speeds(sim, cfg: RunConfig, luggage=frozenset()):
    for agent in sim.agents():
        on_stair = agent.position[1] < 0.0
        speed = cfg.stair_speed if on_stair else cfg.floor_speed
        agent.model.desired_speed = speed * (cfg.luggage_speed_factor if agent.id in luggage else 1.0)


HEADINGS = {"right": (-1.0, 0.0), "down": (0.0, -1.0), "gallery": (1.0, 0.0), "up": (0.0, 1.0),
            "up_right": (0.0, 1.0), "up_left": (0.0, 1.0)}


def _journey(sim, lay: Layout, direction: str, bypass_radius: float, keep_right: bool = False) -> tuple[int, int]:
    """Waypoints in order, then exit: (observed path) bypass left of the column, (keep_right) stair lanes."""
    exit_id = sim.add_exit_stage(exit_zone(lay, "up_right" if direction == "up_left" else direction))
    points = [far_side_waypoint(lay)] if keep_right and direction == "right" else []
    bypass = column_bypass(lay) if direction == "right" else None
    if bypass is not None:
        points.append(bypass)
    if keep_right and direction in ("right", "up_right", "up_left"):
        points += lane_waypoints(lay, direction)
    if not points:
        return sim.add_journey(jps.JourneyDescription([exit_id])), exit_id
    stages = [sim.add_waypoint_stage(p, bypass_radius) for p in points] + [exit_id]
    journey = jps.JourneyDescription(stages)
    for a, b in zip(stages, stages[1:], strict=False):
        journey.set_transition_for_stage(a, jps.Transition.create_fixed_transition(b))
    return sim.add_journey(journey), stages[0]


def _groups(sim, lay: Layout, cfg: RunConfig) -> dict:
    """Per group: journey, first stage, heading and arrival area."""
    dirs = {"main": cfg.direction}
    if cfg.up_share > 0:
        dirs["up"] = "up_right"
    if cfg.up_share > 0 and cfg.luggage_share > 0 and cfg.keep_right:
        dirs["up_bag"] = "up_left"
    groups = {}
    for name, d in dirs.items():
        journey_id, stage_id = _journey(sim, lay, d, cfg.bypass_radius, cfg.keep_right)
        zone_dir = "up" if d in ("up_right", "up_left") else d
        zone = spawn_zone(lay, zone_dir, cfg.spawn_centred, cfg.wide_spawn, cfg.keep_right)
        groups[name] = (journey_id, stage_id, HEADINGS[d], zone)
    return groups


def _arrival_queues(cfg: RunConfig, rng) -> dict[str, list[float]]:
    times = _arrival_times(cfg, rng)
    if cfg.up_share <= 0:
        return {"main": list(times)}
    is_up = rng.random(len(times)) < cfg.up_share
    return {
        "main": [t for t, u in zip(times, is_up, strict=True) if not u],
        "up": [t for t, u in zip(times, is_up, strict=True) if u],
    }


def _spawn_due(sim, cfg, groups, queues, owner, t, rng, luggage):
    for name, queue in queues.items():
        if not queue or queue[0] > t:
            continue
        journey_id, stage_id, heading, zone = groups[name]
        occupied = [a.position for a in sim.agents()]
        p = _free_spot(zone, occupied, rng, 2 * max(cfg.radius, cfg.luggage_radius) + 0.15)
        if p is None:
            continue
        bag = cfg.luggage_share > 0 and rng.random() < cfg.luggage_share
        radius = cfg.luggage_radius if bag else cfg.radius
        if bag and name == "up" and "up_bag" in groups:
            journey_id, stage_id = groups["up_bag"][:2]
        aid = sim.add_agent(_agent_params(cfg, p, journey_id, stage_id, heading, radius))
        owner[aid] = name
        if bag:
            luggage.add(aid)
        queue.pop(0)


def run(lay: Layout, cfg: RunConfig) -> pd.DataFrame:
    """Return trajectories: columns t, id, x, y, vx, vy, group, bag (carries luggage)."""
    rng = np.random.default_rng(cfg.seed)
    sim = jps.Simulation(model=_make_model(cfg), geometry=walkable_area(lay), dt=DT)
    groups = _groups(sim, lay, cfg)
    queues = _arrival_queues(cfg, rng)
    owner: dict[int, str] = {}
    luggage: set[int] = set()
    rows, last_pos, failed = [], {}, False
    while any(queues.values()) or sim.agent_count() > 0:
        t = sim.iteration_count() * DT
        if t > cfg.max_time:
            break
        _spawn_due(sim, cfg, groups, queues, owner, t, rng, luggage)
        _set_speeds(sim, cfg, luggage)
        if sim.iteration_count() % RECORD_EVERY == 0:
            _record(sim, t, rows, last_pos, owner, luggage)
        try:
            sim.iterate()
        except RuntimeError:  # SFM can push an agent through a wall
            failed = True
            break
    df = pd.DataFrame(rows, columns=["t", "id", "x", "y", "vx", "vy", "group", "bag"])
    df.attrs["stuck"] = sim.agent_count() + sum(len(q) for q in queues.values())
    df.attrs["failed"] = failed
    return df


def _record(sim, t, rows, last_pos, owner, luggage):
    dt_rec = RECORD_EVERY * DT
    for a in sim.agents():
        x, y = a.position
        px, py = last_pos.get(a.id, (x, y))
        rows.append((t, a.id, x, y, (x - px) / dt_rec, (y - py) / dt_rec, owner[a.id], a.id in luggage))
        last_pos[a.id] = (x, y)

