"""Backtests and team-level decompositions built on per-shot xFG."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import shrinkage as sh

ZONE_GROUPS = {
    "Restricted Area": "Rim",
    "In The Paint (Non-RA)": "Paint (non-RA)",
    "Mid-Range": "Mid-range",
    "Left Corner 3": "Corner 3",
    "Right Corner 3": "Corner 3",
    "Above the Break 3": "Above-break 3",
    "Backcourt": "Above-break 3",
}
ZONE_ORDER = ["Rim", "Paint (non-RA)", "Mid-range", "Corner 3", "Above-break 3"]


def year_over_year(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Pair each player-season with the same player's next season."""
    seasons = sorted(player_seasons["season"].unique())
    pairs = []
    for cur, nxt in zip(seasons, seasons[1:]):
        a = player_seasons[player_seasons["season"] == cur].set_index("player_id")
        b = player_seasons[player_seasons["season"] == nxt].set_index("player_id")
        pairs.append(a.join(b, lsuffix="_t", rsuffix="_t1", how="inner"))
    return pd.concat(pairs).reset_index()


def backtest(pairs: pd.DataFrame, mu: float, min_fga_t: int, min_fga_t1: int = 100) -> pd.DataFrame:
    """Predict next-season shot-making from this season: league mean vs raw vs shrunk."""
    k = pairs[(pairs["fga_t"] >= min_fga_t) & (pairs["fga_t1"] >= min_fga_t1)]
    target, w = k["raw_t1"], k["fga_t1"]
    preds = {"League average": np.full(len(k), mu), "Raw": k["raw_t"], "Shrunk (EB)": k["eb_t"]}
    rows = []
    for name, p in preds.items():
        rows.append({
            "min_fga": min_fga_t,
            "estimator": name,
            "n_pairs": len(k),
            "wrmse": float(np.sqrt(np.average((target - p) ** 2, weights=w))),
            "corr": float(np.corrcoef(p, target)[0, 1]) if np.std(p) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def half_season_stability(shots: pd.DataFrame, side: str) -> dict[str, float]:
    """Correlate first- vs second-half team metrics; side is 'team' (offense) or 'opponent'."""
    half = (shots.groupby("season")["game_date"].rank(pct=True) > 0.5).astype(int)
    s = sh.summarize(shots.assign(half=half), ["season", side, "half"])
    wide = s.pivot_table(index=["season", side], columns="half", values=["xpps", "raw", "pps"])
    return {m: float(np.corrcoef(wide[m][0], wide[m][1])[0, 1]) for m in ["pps", "xpps", "raw"]}


def team_table(shots: pd.DataFrame) -> pd.DataFrame:
    """Offense and defense split into shot quality (xPPS) and shot-making (PPS - xPPS)."""
    off = sh.summarize(shots, ["team"]).set_index("team")
    dfn = sh.summarize(shots, ["opponent"]).set_index("opponent")
    t = pd.DataFrame({
        "off_pps": off["pps"], "off_xpps": off["xpps"], "off_making": off["raw"],
        "def_pps": dfn["pps"], "def_xpps": dfn["xpps"], "def_making": dfn["raw"],
    })
    t["off_xpps_rank"] = t["off_xpps"].rank(ascending=False).astype(int)
    t["off_making_rank"] = t["off_making"].rank(ascending=False).astype(int)
    t["def_xpps_rank"] = t["def_xpps"].rank(ascending=True).astype(int)
    t["def_making_rank"] = t["def_making"].rank(ascending=True).astype(int)
    return t.rename_axis("team").reset_index()


def zone_profile(shots: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Share of attempts, FG%, xFG% and points per shot by zone group."""
    z = shots.assign(zone=shots["zone_basic"].map(ZONE_GROUPS),
                     pts=shots["made"] * shots["shot_value"],
                     xpts=shots["xfg"] * shots["shot_value"])
    keys = ([by] if by else []) + ["zone"]
    g = z.groupby(keys).agg(fga=("made", "size"), fg=("made", "mean"), xfg=("xfg", "mean"),
                            pps=("pts", "mean"), xpps=("xpts", "mean")).reset_index()
    total = g.groupby(by)["fga"].transform("sum") if by else g["fga"].sum()
    g["share"] = g["fga"] / total
    g["zone"] = pd.Categorical(g["zone"], ZONE_ORDER, ordered=True)
    return g.sort_values(keys).reset_index(drop=True)


def reallocate(profile: pd.DataFrame, moves: dict[tuple[str, str], float]) -> dict[str, float]:
    """Expected points per shot after moving shares of attempts between zones.

    profile: one team's zone_profile rows. moves: {(from_zone, to_zone): share}, where share
    is a fraction of ALL attempts. Each zone keeps its own xPPS, so this prices shot selection
    only - it assumes the extra attempts are of the same quality as the team's current ones.
    """
    p = profile.set_index("zone")
    share = p["share"].astype(float).copy()
    for (src, dst), amount in moves.items():
        if amount > share[src] + 1e-12:
            raise ValueError(f"cannot move {amount:.3f} out of {src} (only {share[src]:.3f})")
        share[src] -= amount
        share[dst] += amount
    before = float((p["share"] * p["xpps"]).sum())
    after = float((share * p["xpps"]).sum())
    return {"xpps_before": before, "xpps_after": after, "delta_per_100": (after - before) * 100}
