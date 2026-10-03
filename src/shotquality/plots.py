"""Figures for the README and report. Static PNGs, light surface."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ef8f8a", "#b8292a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def draw_court(ax, color=AXIS, lw=1.0):
    """Half court in feet, hoop at the origin, baseline at y = -4.75."""
    elements = [
        Circle((0, 0), 0.75, fill=False),
        Rectangle((-3, -1), 6, 0, fill=False),
        Rectangle((-8, -4.75), 16, 19, fill=False),
        Rectangle((-6, -4.75), 12, 19, fill=False),
        Arc((0, 14.25), 12, 12, theta1=0, theta2=180),
        Arc((0, 0), 8, 8, theta1=0, theta2=180),
        Rectangle((-22, -4.75), 0, 13.7, fill=False),
        Rectangle((22, -4.75), 0, 13.7, fill=False),
        Arc((0, 0), 47.5, 47.5, theta1=22.1, theta2=157.9),
        Rectangle((-25, -4.75), 50, 47, fill=False),
    ]
    for e in elements:
        e.set(edgecolor=color, linewidth=lw)
        ax.add_patch(e)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-4.75, 35)
    ax.set_aspect("equal")
    ax.axis("off")


def calibration(cal: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--", label="Perfect calibration")
    ax.errorbar(cal["pred"], cal["actual"], yerr=1.96 * cal["se"], fmt="o", ms=5,
                color=BLUE, ecolor=BLUE, elinewidth=1, capsize=0, label="xFG model (20 bins)")
    ax.set_xlim(0.2, 0.85)
    ax.set_ylim(0.2, 0.85)
    ax.set_xlabel("Predicted make probability")
    ax.set_ylabel("Actual FG%")
    ax.set_title("Calibration on the held-out 2025-26 season")
    ax.legend(loc="upper left", labelcolor=INK_2)
    fig.savefig(path)
    plt.close(fig)


def reliability_curve(stab_n: float, split_half: dict, path):
    n = np.linspace(0, 1500, 300)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(n, n / (n + stab_n), color=BLUE, lw=2, label="Implied by the fitted prior")
    ax.plot(split_half["median_fga"], split_half["full_sample_reliability"], "o", ms=9,
            color=ORANGE, markeredgecolor=SURFACE, markeredgewidth=2,
            label="Measured: odd vs even games")
    ax.axvline(stab_n, color=AXIS, lw=1, ls="--")
    ax.annotate(f"50% signal at ~{stab_n:.0f} FGA", (stab_n, 0.08), xytext=(8, 0),
                textcoords="offset points", color=INK_2)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1500)
    ax.set_xlabel("Field-goal attempts in the sample")
    ax.set_ylabel("Share of variance that is skill")
    ax.set_title("How many shots before shot-making means something?")
    ax.legend(loc="upper left", labelcolor=INK_2)
    fig.savefig(path)
    plt.close(fig)


def backtest_bars(bt: pd.DataFrame, path):
    est = ["League average", "Raw", "Shrunk (EB)"]
    colors = [AXIS, ORANGE, BLUE]
    groups = sorted(bt["min_fga"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 4))
    width = 0.26
    for i, (e, c) in enumerate(zip(est, colors)):
        vals = [bt[(bt.min_fga == g) & (bt.estimator == e)]["wrmse"].iloc[0] * 100
                for g in groups]
        x = np.arange(len(groups)) + (i - 1) * (width + 0.02)
        ax.bar(x, vals, width, color=c, label=e)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(np.arange(len(groups)), [f"≥{g} FGA in year t" for g in groups])
    ax.set_ylabel("Next-season error (pts per 100 shots, weighted RMSE)")
    ax.set_title("Predicting next season's shot-making")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=3, labelcolor=INK_2, bbox_to_anchor=(1, 1.02))
    ax.set_ylim(0, max(bt["wrmse"]) * 100 * 1.25)
    fig.savefig(path)
    plt.close(fig)


def place_labels(ax, points, avoid=(), **text_kw):
    """Greedy label placement: try offsets around each point, keep the first that collides
    with no earlier label or marker (including unlabeled `avoid` points). Falls back to the
    default offset."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    candidates = [(5, 3, "left", "bottom"), (5, -3, "left", "top"), (-5, 3, "right", "bottom"),
                  (-5, -3, "right", "top"), (0, 7, "center", "bottom"), (0, -7, "center", "top")]
    marker_boxes = []
    for x, y in avoid:
        cx, cy = ax.transData.transform((x, y))
        marker_boxes.append((cx - 7, cy - 7, cx + 7, cy + 7))
    for _, x, y in points:
        cx, cy = ax.transData.transform((x, y))
        marker_boxes.append((cx - 5, cy - 5, cx + 5, cy + 5))
    placed = []

    def overlaps(b, boxes):
        return any(b[0] < o[2] and b[2] > o[0] and b[1] < o[3] and b[3] > o[1] for o in boxes)

    n_avoid = len(avoid)
    for i, (label, x, y) in enumerate(points):
        j = i + n_avoid
        others = marker_boxes[:j] + marker_boxes[j + 1:]
        for k, (dx, dy, ha, va) in enumerate(candidates):
            t = ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, zorder=2, **text_kw)
            e = t.get_window_extent(renderer)
            box = (e.x0, e.y0, e.x1, e.y1)
            if not overlaps(box, placed + others) or k == len(candidates) - 1:
                if overlaps(box, placed + others):
                    t.remove()
                    t = ax.annotate(label, (x, y), xytext=candidates[0][:2],
                                    textcoords="offset points", zorder=2, **text_kw)
                    e = t.get_window_extent(renderer)
                    box = (e.x0, e.y0, e.x1, e.y1)
                placed.append(box)
                break
            t.remove()


def team_quadrant(teams: pd.DataFrame, side: str, highlight: str, path, label_offset=(6, 4)):
    """x = shot quality (xPPS), y = shot-making; for defense both axes are 'allowed'.

    label_offset places the highlighted team's label (points); far offsets get a leader line.
    """
    x, y = teams[f"{side}_xpps"], teams[f"{side}_making"] * 100
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.axvline(x.mean(), color=AXIS, lw=1)
    ax.axhline(0, color=AXIS, lw=1)
    hl = teams["team"] == highlight
    ax.scatter(x[~hl], y[~hl], s=46, color=MUTED, alpha=0.75, edgecolor=SURFACE, lw=1.5)
    ax.scatter(x[hl], y[hl], s=90, color=ORANGE, edgecolor=SURFACE, lw=2, zorder=3)
    pts = [(r["team"], r[f"{side}_xpps"], r[f"{side}_making"] * 100) for _, r in teams.iterrows()]
    hl_pt = next(p for p in pts if p[0] == highlight)
    place_labels(ax, [p for p in pts if p[0] != highlight], avoid=[hl_pt[1:]],
                 fontsize=8, color=MUTED)
    ax.annotate(highlight, hl_pt[1:], xytext=label_offset, textcoords="offset points",
                fontsize=10, color=INK, fontweight="bold", zorder=5,
                path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)],
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8, shrinkB=6)
                if max(map(abs, label_offset)) > 12 else None)
    if side == "off":
        ax.set_xlabel("Shot quality: expected points per shot (xPPS) →  better")
        ax.set_ylabel("Shot-making: points above expected per 100 shots")
        ax.set_title("2025-26 offense: shot quality vs shot-making")
    else:
        ax.set_xlabel("Shot quality allowed: opponent xPPS  (← better defense)")
        ax.set_ylabel("Opponent shot-making allowed, per 100 shots")
        ax.set_title("2025-26 defense: quality allowed vs opponent shot-making")
    fig.savefig(path)
    plt.close(fig)


def shot_diet_map(team_shots: pd.DataFrame, league_shots: pd.DataFrame, title: str, path):
    """Hex map of a team's shot frequency minus the league's, per 100 attempts."""
    kw = dict(gridsize=(16, 10), extent=(-25, 25, -4.75, 30))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    lg = ax.hexbin(league_shots["x_ft"], league_shots["y_ft"], **kw)
    tm = ax.hexbin(team_shots["x_ft"], team_shots["y_ft"], **kw)
    lg_share = lg.get_array() / len(league_shots)
    diff = (tm.get_array() / len(team_shots) - lg_share) * 100
    lg.remove()
    tm.set_array(np.ma.masked_where(lg_share < 0.0008, np.asarray(diff)))
    lim = np.percentile(np.abs(np.asarray(diff)[lg_share >= 0.0008]), 97)
    tm.set_cmap(DIVERGING)
    tm.set_clim(-lim, lim)
    tm.set_edgecolor(SURFACE)
    tm.set_linewidth(0.8)
    draw_court(ax)
    cb = fig.colorbar(tm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Attempts per 100 vs league average", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def player_intervals(players: pd.DataFrame, title: str, path):
    p = players.sort_values("eb")
    y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(6.6, 0.38 * len(p) + 1.4))
    ax.axvline(0, color=AXIS, lw=1)
    ax.hlines(y, p["eb_lo"] * 100, p["eb_hi"] * 100, color=BLUE, lw=2.5, alpha=0.35)
    ax.plot(p["raw"] * 100, y, "o", ms=6, mfc=SURFACE, mec=ORANGE, mew=1.5, label="Raw")
    ax.plot(p["eb"] * 100, y, "o", ms=7, color=BLUE, mec=SURFACE, mew=1.5,
            label="Shrunk estimate, 90% interval")
    ax.set_yticks(y, [f"{n}  ({int(f)} FGA)" for n, f in zip(p["player"], p["fga"])])
    ax.tick_params(axis="y", colors=INK_2, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Shot-making: points above expected per 100 shots")
    ax.set_title(title)
    ax.legend(loc="upper left", labelcolor=INK_2)
    fig.savefig(path)
    plt.close(fig)
