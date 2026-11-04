"""Animated clips of the stair head, written as MP4 and GIF to media/.

Usage: .venv/bin/python scripts/make_media.py [clip ...]
Clips: column, speed, models, crowd, counterflow, luggage (default: all).
Frames are drawn with matplotlib on the dark outreach background (#0A0E14)
and encoded with ffmpeg (palette pass for the GIF).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

import kopenhagen.simulate as sim
from kopenhagen.geometry import SCENARIOS, column, glass, walkable_area
from kopenhagen.metrics import BRUSH, HEAD_ZONE, clearance_to_glass

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "media"
BG = "#0A0E14"
FLOOR = "#1b2330"
STAIR = "#3b3129"
GLASS = "#6fa8dc"
AGENT = "#f2f2f2"
UP = "#f2b84b"  # people coming up the stair
LUGGAGE_RADIUS = 0.30  # matches RunConfig.luggage_radius
HOT = "#ff4d4d"
FPS = 20
TAIL = 2.0  # seconds of trail behind each person
LOGO = ROOT / "assets" / "jupedsim_logo.png"  # rasterised from scenarios-gif/public/jupedsim_logo.svg


def white_logo() -> np.ndarray:
    """Logo in white (as in the Remotion projects), keeping its alpha."""
    rgba = plt.imread(LOGO)
    rgba[..., :3] = 1.0
    rgba[..., 3] *= 0.85
    return rgba


CROWD = {"wide_spawn": True, "bypass_radius": 0.8, "max_time": 400.0}


def simulate(lay, record_every=5, n_agents=6, inflow=0.5, seed=3, **kw) -> pd.DataFrame:
    """One frame per recorded sample: record_every=5 is real time at 20 fps, 10 is 2x."""
    sim.RECORD_EVERY = record_every
    cfg = sim.RunConfig(n_agents=n_agents, inflow=inflow, seed=seed, **kw)
    traj = sim.run(lay, cfg)
    traj = clearance_to_glass(lay, traj, cfg.radius)
    in_head = (traj["y"] > HEAD_ZONE[0]) & (traj["y"] < HEAD_ZONE[1])
    traj["hot"] = in_head & (traj["clr_glass"] < BRUSH)
    traj["frame"] = np.round(traj["t"] / (record_every * sim.DT)).astype(int)
    traj.attrs["speedup"] = record_every * sim.DT * FPS
    count = traj.groupby("frame").size()
    traj.attrs["end"] = int(count.index[count.diff().fillna(0) != 0].max())  # last arrival/exit
    return traj


def draw_scene(ax, lay):
    ax.set_facecolor(BG)
    area = walkable_area(lay)
    ax.fill(*area.exterior.xy, color=FLOOR, zorder=0)
    ax.fill_between([lay.x_left, lay.x_right], -lay.stair_length, 0, color=STAIR, zorder=0)
    for y in np.arange(-lay.stair_length, 0, 0.3):
        ax.plot([lay.x_left, lay.x_right], [y, y], color=BG, lw=0.8, zorder=1)
    col = column(lay)
    if col is not None:
        ax.add_patch(Circle((col.centroid.x, col.centroid.y), lay.column_diameter / 2,
                            facecolor="#4a5563", edgecolor="#c0c6cc", lw=1.5, zorder=5))
    for line in glass(lay).values():
        ax.plot(*line.xy, color=GLASS, lw=3, solid_capstyle="butt", zorder=4)
    ax.set_xlim(-2.0, 4.6)
    ax.set_ylim(-3.0, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_people(ax, lay, traj, frame, radius=0.25):
    now = traj[traj["frame"] == frame]
    recent = traj[(traj["frame"] <= frame) & (traj["frame"] > frame - FPS // 2)]
    ring = HOT if recent["hot"].any() else "#9aa3ad"
    ax.add_patch(Circle(lay.inner_corner, 0.18, fill=False, edgecolor=ring, lw=2, zorder=8))
    tail = traj[(traj["frame"] <= frame) & (traj["frame"] > frame - TAIL * FPS)]
    for _, g in tail.groupby("id"):
        ax.plot(g["x"], g["y"], color=AGENT, lw=1.2, alpha=0.35, zorder=6)
    for _, row in now.iterrows():
        color = HOT if row["hot"] else (UP if row["group"] == "up" else AGENT)
        r = LUGGAGE_RADIUS if row["bag"] else radius
        ax.add_patch(Circle((row["x"], row["y"]), r, color=color, zorder=7))
        if row["bag"]:
            ax.add_patch(Circle((row["x"], row["y"]), 0.09, color=BG, zorder=7.5))


def render(panels, title, name, legend="red = closer than 10 cm to the glass", gif_scale="2/3"):
    """panels: list of (label, layout, trajectory)."""
    speedup = panels[0][2].attrs["speedup"]
    play = f"   {speedup:.0f}x speed" if speedup > 1 else ""
    n_frames = max(p[2].attrs["end"] for p in panels) + FPS  # stop when only stuck people remain
    logo = white_logo()
    tmp = Path(tempfile.mkdtemp())
    for f in range(n_frames):
        fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 5.0), facecolor=BG)
        axes = np.atleast_1d(axes)
        for ax, (label, lay, traj) in zip(axes, panels, strict=True):
            draw_scene(ax, lay)
            draw_people(ax, lay, traj, f)
            ax.set_title(label, color="white", fontsize=12, loc="left", pad=6)
        fig.suptitle(title, color="white", fontsize=13, x=0.03, ha="left")
        fig.text(0.03, 0.025, f"t = {f * speedup / FPS:5.1f} s{play}   {legend}",
                 color="#9aa3ad", fontsize=9)
        width = 1.1 / fig.get_size_inches()[0]  # about 1.1 inch wide in every clip
        height = width * logo.shape[0] / logo.shape[1] * fig.get_size_inches()[0] / fig.get_size_inches()[1]
        logo_ax = fig.add_axes((0.98 - width, 0.015, width, height))
        logo_ax.imshow(logo)
        logo_ax.axis("off")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.07, wspace=0.05)
        fig.savefig(tmp / f"f{f:05d}.png", dpi=150, facecolor=BG)
        plt.close(fig)
    encode(tmp, name, gif_scale)
    shutil.rmtree(tmp)


def encode(frames: Path, name: str, gif_scale: str = "2/3"):
    OUT.mkdir(exist_ok=True)
    src = ["-framerate", str(FPS), "-i", str(frames / "f%05d.png")]
    even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-vf", even, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-bf", "0", str(OUT / f"{name}.mp4")], check=True)
    pal = frames / "palette.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-vf",
                    f"fps=15,scale=iw*{gif_scale}:-1:flags=lanczos,palettegen=max_colors=64", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-i", str(pal), "-lavfi",
                    f"fps=15,scale=iw*{gif_scale}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none", str(OUT / f"{name}.gif")], check=True)


def clip_column():
    panels = [(name, SCENARIOS[name], simulate(SCENARIOS[name], floor_speed=1.8))
              for name in ("as built", "no column")]
    render(panels, "Walking fast (1.8 m/s) onto the stair", "stair_column_vs_none")


def clip_speed():
    lay = SCENARIOS["as built"]
    panels = [(f"{v} m/s", lay, simulate(lay, floor_speed=v)) for v in (1.2, 1.8)]
    render(panels, "Same stair, normal pace vs hurrying", "stair_speed")


def clip_models():
    lay = SCENARIOS["as built"]
    labels = {"sfm": "Social force", "warp": "WarpDriver", "cfsm": "Collision-free speed"}
    panels = [(label, lay, simulate(lay, model=m, floor_speed=1.8)) for m, label in labels.items()]
    render(panels, "Three JuPedSim models, 1.8 m/s, as built", "stair_three_models")


def clip_crowd():
    panels = [(name, SCENARIOS[name], simulate(SCENARIOS[name], record_every=10, n_agents=80,
                                               inflow=1.5, model="sfm", **CROWD))
              for name in ("as built", "no column")]
    render(panels, "A plane unloads: 80 people at 1.5 per second", "stair_crowd", gif_scale="1/2")


def clip_luggage():
    panels = [(name, SCENARIOS[name], simulate(SCENARIOS[name], record_every=10, n_agents=40, inflow=1.5,
                                               up_share=0.5, luggage_share=0.5, model="cfsm", keep_right=True,
                                               seed=9, **CROWD))
              for name in ("as built", "no column")]
    render(panels, "40 people, half coming up, half with luggage", "stair_luggage",
           legend="amber = coming up   dot = luggage   red = < 10 cm to glass")


def clip_counterflow():
    panels = [(name, SCENARIOS[name], simulate(SCENARIOS[name], n_agents=20, inflow=0.8, up_share=0.5,
                                               model="cfsm", keep_right=True, seed=0, **CROWD))
              for name in ("as built", "no column")]
    render(panels, "20 people keeping right, half coming up (collision-free speed model)", "stair_counterflow_v2",
           legend="amber = coming up   red = closer than 10 cm to the glass")


CLIPS = {"column": clip_column, "speed": clip_speed, "models": clip_models,
         "crowd": clip_crowd, "counterflow": clip_counterflow, "luggage": clip_luggage}


def main():
    for name in sys.argv[1:] or CLIPS:
        CLIPS[name]()


if __name__ == "__main__":
    main()
