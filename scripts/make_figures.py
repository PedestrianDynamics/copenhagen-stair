"""Make the study figures from results/*.csv.

Usage: .venv/bin/python scripts/make_figures.py
House style without seaborn: whitegrid look, DejaVu Sans, dimgrey text, no spines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from PIL import Image, ImageOps

from kopenhagen.geometry import SCENARIOS, Layout, column, gaps, glass, walkable_area

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
GLASS = "#4575b4"
ACCENT = "#bd0c0c"
GREY = "#7f7f7f"
ORDER = ["as built", "no column", "column set back 2 m", "column against right glass"]
MARKERS = {"as built": "o", "no column": "s", "column set back 2 m": "^", "column against right glass": "D"}
COLORS = {"as built": ACCENT, "no column": "#1f253f", "column set back 2 m": "#fc8d59", "column against right glass": "#72a5b4"}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": "white",
        "axes.edgecolor": "white",
        "axes.grid": False,
        "axes.labelcolor": "dimgrey",
        "axes.titlecolor": "#333333",
        "xtick.color": "dimgrey",
        "ytick.color": "dimgrey",
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.framealpha": 0.8,
        "legend.edgecolor": "lightgrey",
        "legend.labelcolor": "dimgrey",
    })


def despine(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_plan(ax, lay: Layout, floor="#eeeeee"):
    area = walkable_area(lay)
    ax.fill(*area.exterior.xy, color=floor, zorder=0)
    ax.fill_between([lay.x_left, lay.x_right], -lay.stair_length, 0, color="#e2d6c8", zorder=0)
    for y in np.arange(-lay.stair_length, 0, 0.3):
        ax.plot([lay.x_left, lay.x_right], [y, y], color="white", lw=0.6, zorder=1)
    col = column(lay)
    if col is not None:
        cx, cy = col.centroid.x, col.centroid.y
        ax.add_patch(Circle((cx, cy), lay.column_diameter / 2, color="#444444", zorder=5))
    for line in glass(lay).values():
        ax.plot(*line.xy, color=GLASS, lw=3, solid_capstyle="butt", zorder=4)
    ax.set_aspect("equal")
    despine(ax)


MY_PATH = [(4.5, 3.05), (1.0, 3.05), (-0.2, 2.9), (-0.55, 2.4), (-0.65, 1.5), (-0.6, 0.9),
           (-0.4, 0.4), (-0.2, 0.0), (0.1, -0.7)]


def fig_plan():
    lay = SCENARIOS["as built"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 6.5), gridspec_kw={"width_ratios": [1, 1.5]})
    photo = ROOT / "photos" / "stair_from_below.jpg"
    a0.axis("off")
    if photo.exists():
        img = ImageOps.exif_transpose(Image.open(photo))
        img.thumbnail((1400, 1400))
        a0.imshow(img)
        a0.set_title("The stair, seen from below", loc="left", fontsize=12)

    draw_plan(a1, lay)
    g = gaps(lay)
    r = lay.column_diameter / 2
    cx, cy = lay.column_offset, lay.column_setback
    a1.annotate("", xy=(lay.x_right, cy), xytext=(cx + r, cy),
                arrowprops={"arrowstyle": "<->", "color": "dimgrey", "lw": 1.0})
    a1.text(1.15, cy, f"{g['column_to_right_glass']:.2f} m\ncolumn to glass", color="dimgrey", va="center", fontsize=9)
    a1.annotate("", xy=(cx, 0), xytext=(cx, cy - r), arrowprops={"arrowstyle": "<->", "color": "dimgrey"})
    a1.text(cx + 0.08, 0.32, f"{g['column_to_nosing']:.2f} m", color="dimgrey", fontsize=9)
    a1.annotate("", xy=(lay.x_left, -2.5), xytext=(lay.x_right, -2.5), arrowprops={"arrowstyle": "<->", "color": "dimgrey"})
    a1.text(0, -2.38, f"{lay.stair_width:.1f} m", color="dimgrey", ha="center", fontsize=9)
    a1.text(1.15, cy + 0.75, f"column ø {lay.column_diameter:.1f} m", color="dimgrey", fontsize=9)
    a1.plot(*lay.inner_corner, marker="o", ms=9, mfc="none", mec=ACCENT, mew=1.5, zorder=6)
    a1.text(lay.x_left - 0.12, -0.35, "end of the\nleft glass", color=ACCENT, ha="right", fontsize=9)
    px, py = zip(*MY_PATH, strict=True)
    a1.plot(px, py, color=ACCENT, lw=2.5, alpha=0.8, zorder=7, solid_capstyle="round")
    a1.annotate("", xy=MY_PATH[-1], xytext=MY_PATH[-2],
                arrowprops={"arrowstyle": "-|>", "color": ACCENT, "lw": 2.5, "mutation_scale": 18}, zorder=7)
    a1.text(3.0, 3.35, "the route walked", color=ACCENT, ha="center", fontsize=9)
    a1.text(0, -3.7, "stair down", color="dimgrey", ha="center", fontsize=9)
    a1.text(-3.0, 0.9, "gallery", color="dimgrey", ha="center", fontsize=9)
    a1.text(3.0, 1.2, "void", color="dimgrey", ha="center", fontsize=9)
    a1.plot([], [], color=GLASS, lw=3, label="glass balustrade")
    a1.legend(loc="lower left", fontsize=9)
    a1.set_xlim(-4, 4.6)
    a1.set_ylim(-4, 5)
    a1.set_xlabel("m")
    a1.set_title("Plan used in the simulation (estimated from photos)", loc="left", fontsize=12)
    save(fig, "fig1_plan")


def fig_trajectories():
    traj = pd.read_csv(RES / "example_trajectories.csv")
    fig, axes = plt.subplots(2, 4, figsize=(15, 9), sharex=True, sharey=True)
    for row, speed in enumerate((1.2, 1.8)):
        for ax, name in zip(axes[row], ORDER, strict=True):
            lay = SCENARIOS[name]
            draw_plan(ax, lay)
            sub = traj[(traj["scenario"] == name) & (traj["speed"] == speed)]
            for _, g in sub.groupby("id"):
                ax.plot(g["x"], g["y"], color="#1f253f", lw=0.7, alpha=0.5, zorder=3)
            ax.plot(*lay.inner_corner, marker="o", ms=10, mfc="none", mec=ACCENT, mew=1.5, zorder=6)
            ax.set_xlim(-2.0, 4.6)
            ax.set_ylim(-3.0, 4.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name, loc="left", fontsize=11)
        axes[row, 0].set_ylabel(f"walking at {speed} m/s", fontsize=11)
    axes[1, 0].annotate("fast: people swing\nout to the end\nof the left glass", xy=(-0.95, 0.15), xytext=(-1.95, 3.7),
                        color=ACCENT, fontsize=9, arrowprops={"arrowstyle": "->", "color": ACCENT})
    fig.suptitle("20 people, one at a time, on the path from the sketch (social force model, one seed)",
                 x=0.06, ha="left", fontsize=12, color="#333333")
    fig.tight_layout()
    save(fig, "fig2_trajectories")


def fig_speed():
    ag = pd.read_csv(RES / "speed_agents.csv")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6))
    series = [("sfm", name) for name in ORDER] + [("cfsm", "as built")]
    for model, name in series:
        sub = ag[(ag["model"] == model) & (ag["scenario"] == name)]
        clr = sub.groupby("speed")["min_clr"].median() * 100
        ang = sub.groupby("speed")["entry_angle"].median()
        kw = {"color": COLORS[name], "marker": MARKERS[name], "ms": 5, "lw": 1.6}
        label = name
        if model == "cfsm":
            kw.update(color=GREY, ls="--", marker="x")
            label = "as built, model without inertia"
        a0.plot(clr.index, clr.values, label=label, **kw)
        a1.plot(ang.index, ang.values, **kw)
    for ax in (a0, a1):
        ax.axvspan(1.2, 1.4, color="#eaeaf2", zorder=0)
        ax.set_xlabel("walking speed on the floor (m/s)")
        ax.grid(alpha=0.7, lw=1, axis="y", color="#eaeaf2")
        despine(ax)
    a0.text(1.3, 2, "normal\nwalking", ha="center", color=GREY, fontsize=8)
    a0.axhline(10, color=GREY, ls=":", lw=1)
    a0.set_ylim(0, 75)
    a0.set_ylabel("closest gap to the glass (cm, median)")
    a0.set_title("How close you get to the glass", loc="left", fontsize=11)
    handles, labels = a0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    a1.set_ylim(0, 45)
    a1.set_ylabel("angle to the stair axis (deg, median)")
    a1.set_title("How crooked you step onto the stair", loc="left", fontsize=11)
    a1.annotate("with the column you reach the\ntop step still turning", xy=(1.8, 36), xytext=(1.0, 41.5),
                color="dimgrey", fontsize=9, arrowprops={"arrowstyle": "->", "color": "dimgrey"})
    fig.suptitle("People walking alone on the sketched path; 8 seeds × 20 people per point",
                 x=0.06, ha="left", fontsize=11, color="#333333")
    fig.tight_layout()
    save(fig, "fig3_speed")


MODEL_STYLE = {
    "sfm": ("Social force (inertia)", ACCENT, "o"),
    "warp": ("WarpDriver (anticipation)", "#fc8d59", "^"),
    "cfsm": ("Collision-free speed (no inertia)", "#4575b4", "s"),
}


def fig_models():
    traj = pd.read_csv(RES / "model_trajectories.csv")
    ag = pd.read_csv(RES / "speed_agents.csv")
    ag = ag[ag["scenario"].isin(["as built", "no column"])]
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1])
    lay = SCENARIOS["as built"]
    for i, (model, (label, color, _)) in enumerate(MODEL_STYLE.items()):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        draw_plan(ax, lay)
        sub = traj[(traj["model"] == model) & (traj["scenario"] == "as built")]
        for _, g in sub.groupby("id"):
            ax.plot(g["x"], g["y"], color=color, lw=0.8, alpha=0.6, zorder=3)
        ax.plot(*lay.inner_corner, marker="o", ms=10, mfc="none", mec=ACCENT, mew=1.5, zorder=6)
        ax.set_xlim(-2.0, 4.6)
        ax.set_ylim(-3.0, 4.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label}\nas built, 1.8 m/s", loc="left", fontsize=11)
    a0 = fig.add_subplot(gs[1, :3])
    a1 = fig.add_subplot(gs[1, 3:])
    for model, (label, color, marker) in MODEL_STYLE.items():
        for name, ls, mfc in (("as built", "-", color), ("no column", "--", "white")):
            sub = ag[(ag["model"] == model) & (ag["scenario"] == name)]
            clr = sub.groupby("speed")["min_clr"].median() * 100
            ang = sub.groupby("speed")["entry_angle"].median()
            kw = {"color": color, "marker": marker, "ms": 6, "lw": 1.6, "ls": ls, "mfc": mfc}
            a0.plot(clr.index, clr.values, label=f"{label}, {name}", **kw)
            a1.plot(ang.index, ang.values, **kw)
    for ax in (a0, a1):
        ax.axvspan(1.2, 1.4, color="#eaeaf2", zorder=0)
        ax.set_xlabel("walking speed on the floor (m/s)")
        ax.grid(alpha=0.7, lw=1, axis="y", color="#eaeaf2")
        despine(ax)
    a0.axhline(10, color=GREY, ls=":", lw=1)
    a0.set_ylim(0, 75)
    a0.set_ylabel("closest gap to the glass (cm, median)")
    a0.set_title("How close you get to the glass", loc="left", fontsize=11)
    a1.set_ylim(0, 45)
    a1.set_ylabel("angle to the stair axis (deg, median)")
    a1.set_title("How crooked you step onto the stair", loc="left", fontsize=11)
    a0.annotate("only the model with inertia\ngets worse when you hurry", xy=(1.9, 8.5), xytext=(1.3, 40),
                color="dimgrey", fontsize=9, arrowprops={"arrowstyle": "->", "color": "dimgrey"})
    handles, labels = a0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Same stair, same route, three JuPedSim models (solid = as built, dashed = no column; 8 seeds × 20 people)",
                 x=0.06, ha="left", fontsize=12, color="#333333")
    fig.tight_layout()
    save(fig, "fig5_models")


def fig_crowd():
    runs = pd.read_csv(RES / "crowd_runs.csv")
    panels = (("clear_time", "time until all 120 are down (s)", "How long the plane takes to clear"),
              ("flow_down", "flow over the top step (persons/s)", "How many people get onto the stair per second"),
              ("travel_down", "time per person, median (s)", "How long each person needs"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (col, ylabel, title) in zip(axes, panels, strict=True):
        for model, (label, color, marker) in MODEL_STYLE.items():
            for name, ls, mfc in (("as built", "-", color), ("no column", "--", "white")):
                g = runs[(runs["model"] == model) & (runs["scenario"] == name)].groupby("inflow")[col]
                m = g.mean()
                ax.errorbar(m.index, m.values, yerr=[m - g.min(), g.max() - m], color=color, marker=marker,
                            ms=6, lw=1.6, ls=ls, mfc=mfc, capsize=3, label=f"{label}, {name}")
        ax.set_xticks([1.0, 1.5, 2.0])
        ax.set_xlabel("arrival rate (persons/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.7, lw=1, axis="y", color="#eaeaf2")
        despine(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("A plane unloads: 120 people walking down (5 seeds, bars = min–max; solid = as built, dashed = no column)",
                 x=0.06, ha="left", fontsize=12, color="#333333")
    fig.tight_layout()
    save(fig, "fig6_crowd")


def fig_sweep():
    runs = pd.read_csv(RES / "sweep_runs.csv")
    base = Layout()
    titles = {
        "column_setback": ("column set back from top step (m)", base.column_setback),
        "column_offset": ("column offset from stair axis (m, + = towards right glass)", base.column_offset),
        "stair_width": ("stair width (m)", base.stair_width),
        "column_diameter": ("column diameter (m)", base.column_diameter),
        "glass_end": ("where the right glass ends (m past top step)", base.glass_end),
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    axes[1, 2].axis("off")
    for ax, (param, (label, guess)) in zip(axes.flat, titles.items(), strict=False):
        sub = runs[runs["param"] == param]
        for has_col, color, marker, name in ((True, ACCENT, "o", "with column"), (False, "#1f253f", "s", "no column")):
            g = sub[sub["has_column"] == has_col].groupby("value")["median_min_clr"]
            m, lo, hi = g.mean(), g.min(), g.max()
            ax.fill_between(m.index, lo, hi, color=color, alpha=0.12, lw=0)
            ax.plot(m.index, m.values, color=color, marker=marker, ms=5, lw=1.5, label=name)
        ax.axvline(guess, color=GREY, ls=":", lw=1)
        ax.text(guess, 0.02, " my estimate", color=GREY, fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylim(0, 0.85)
        ax.grid(alpha=0.7, lw=1, axis="y", color="#eaeaf2")
        despine(ax)
    axes[0, 0].set_ylabel("median closest gap to glass (m)")
    axes[1, 0].set_ylabel("median closest gap to glass (m)")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    axes[0, 1].annotate("flush with the right glass:\nhalf the effect", xy=(0.75, 0.32), xytext=(-0.1, 0.5),
                        color="dimgrey", fontsize=9, arrowprops={"arrowstyle": "->", "color": "dimgrey"})
    fig.suptitle("Does it depend on my guessed dimensions? (social force model, 8 seeds, band = min–max)",
                 x=0.06, ha="left", fontsize=11, color="#333333")
    fig.tight_layout()
    save(fig, "fig4_sensitivity")


def main():
    style()
    fig_plan()
    fig_trajectories()
    fig_speed()
    fig_models()
    fig_crowd()
    fig_sweep()


if __name__ == "__main__":
    main()
