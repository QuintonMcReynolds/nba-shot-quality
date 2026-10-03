"""Empirical-Bayes shrinkage of shot-making (make rate above expected).

For a group g (a player, a team's defense, ...) with n shots, the per-shot
residual r = value * (made - xFG) is points scored above expectation, and m_g is
its mean. Given the true skill theta_g, m_g is approximately Normal(theta_g, v_g)
with v_g = sum(value^2 * xFG * (1 - xFG)) / n^2 - the binomial noise implied by
the shots' own difficulty. Skills are drawn from
Normal(mu, tau^2) across the league. tau^2 is estimated by the method of
moments, and each group's estimate is pulled toward mu by B = tau^2 / (tau^2 + v_g).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(shots: pd.DataFrame, by: list[str], xfg: str = "xfg") -> pd.DataFrame:
    """Per-group shots, makes, expected points, points above expected per shot, and its variance."""
    value = shots["shot_value"]
    g = shots.assign(
        _pts=value * shots["made"],
        _xpts=value * shots[xfg],
        _r=value * (shots["made"] - shots[xfg]),
        _v=value**2 * shots[xfg] * (1 - shots[xfg]),
    )
    out = g.groupby(by, observed=True).agg(
        fga=("made", "size"),
        fgm=("made", "sum"),
        xfgm=(xfg, "sum"),
        pts=("_pts", "sum"),
        xpts=("_xpts", "sum"),
        resid_sum=("_r", "sum"),
        var_sum=("_v", "sum"),
    ).reset_index()
    out["fg_pct"] = out["fgm"] / out["fga"]
    out["xfg_pct"] = out["xfgm"] / out["fga"]
    out["pps"] = out["pts"] / out["fga"]
    out["xpps"] = out["xpts"] / out["fga"]
    out["raw"] = out["resid_sum"] / out["fga"]
    out["raw_var"] = out["var_sum"] / out["fga"] ** 2
    return out


def fit_prior(summary: pd.DataFrame, min_fga: int = 50) -> tuple[float, float]:
    """Method-of-moments estimate of (mu, tau^2) from groups with at least min_fga shots."""
    s = summary[summary["fga"] >= min_fga]
    w = s["fga"] / s["fga"].sum()
    mu = float(np.sum(w * s["raw"]))
    tau2 = float(np.sum(w * ((s["raw"] - mu) ** 2 - s["raw_var"])))
    return mu, max(tau2, 1e-8)


def shrink(summary: pd.DataFrame, mu: float, tau2: float) -> pd.DataFrame:
    out = summary.copy()
    b = tau2 / (tau2 + out["raw_var"])
    out["reliability"] = b
    out["eb"] = mu + b * (out["raw"] - mu)
    out["eb_sd"] = np.sqrt(b * out["raw_var"])
    out["eb_lo"] = out["eb"] - 1.645 * out["eb_sd"]  # 90% interval
    out["eb_hi"] = out["eb"] + 1.645 * out["eb_sd"]
    return out


def stabilization_point(shots: pd.DataFrame, tau2: float, xfg: str = "xfg") -> float:
    """Shots needed before an estimate is 50% signal (B = 0.5): mean per-shot variance / tau^2."""
    p = shots[xfg]
    return float(np.mean(shots["shot_value"] ** 2 * p * (1 - p)) / tau2)


def split_half_reliability(
    shots: pd.DataFrame, by: list[str], min_fga: int = 200, xfg: str = "xfg"
) -> dict[str, float]:
    """Correlate odd-game vs even-game residuals; Spearman-Brown up to the full sample."""
    halves = shots.assign(half=shots["game_id"] % 2)
    s = summarize(halves, by + ["half"], xfg)
    wide = s.pivot_table(index=by, columns="half", values=["raw", "fga"])
    wide = wide[(wide["fga"][0] + wide["fga"][1]) >= min_fga].dropna()
    r = float(np.corrcoef(wide["raw"][0], wide["raw"][1])[0, 1])
    return {
        "groups": int(len(wide)),
        "median_fga": float((wide["fga"][0] + wide["fga"][1]).median()),
        "split_half_r": r,
        "full_sample_reliability": 2 * r / (1 + r),
    }
