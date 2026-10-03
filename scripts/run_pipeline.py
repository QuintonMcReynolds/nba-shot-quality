"""Run the full analysis: data -> xFG models -> shrinkage -> backtests -> team/Suns outputs.

Usage: python scripts/run_pipeline.py [--team PHX]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from shotquality import analysis, data, model, plots
from shotquality import shrinkage as sh

ROOT = Path(__file__).resolve().parents[1]
ART, DOCS, PROC = ROOT / "artifacts", ROOT / "docs", ROOT / "data" / "processed"
TEST_SEASON = "2025-26"


def main(team: str) -> None:
    for d in (ART, DOCS, PROC):
        d.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}

    shots = data.load()
    metrics["data"] = shots.groupby("season").agg(
        fga=("made", "size"), games=("game_id", "nunique"), fg_pct=("made", "mean"),
        three_rate=("is_three", "mean")).round(4).to_dict(orient="index")
    print(f"{len(shots):,} shots loaded")

    # 1. Shot-quality model: train on earlier seasons, test on the latest one.
    train, test = shots[shots.season != TEST_SEASON], shots[shots.season == TEST_SEASON]
    y = test["made"].to_numpy()
    candidates = {
        "League average": None,
        "Zone average (NBA shot-chart zones)": model.ZoneBaseline().fit(train),
        **{f"GBM: {k}": model.XFGModel(f).fit(train) for k, f in model.FEATURE_SETS.items()},
    }
    comparison = {}
    for name, m in candidates.items():
        p = [train["made"].mean()] * len(y) if m is None else m.predict_proba(test)
        comparison[name] = {k: round(v, 4) for k, v in model.evaluate(y, p).items()}
        print(f"  {name:40s} {comparison[name]}")
    metrics["holdout_model_comparison"] = comparison
    best = candidates["GBM: location+context"]
    cal = model.calibration_table(y, best.predict_proba(test))
    plots.calibration(cal, DOCS / "calibration.png")

    # 2. Out-of-season xFG for every shot, so player-seasons are never scored in-sample.
    shots["xfg"] = model.out_of_season_predictions(shots, model.FEATURE_SETS["location+context"])
    shots.to_parquet(PROC / "shots_xfg.parquet")

    # 3. Shot-making with empirical-Bayes shrinkage.
    ps = sh.summarize(shots, ["season", "player_id", "player"])
    mu, tau2 = sh.fit_prior(ps, min_fga=100)
    ps = sh.shrink(ps, mu, tau2)
    stab = sh.stabilization_point(shots, tau2)
    split = sh.split_half_reliability(shots, ["season", "player_id"], min_fga=300)
    by_type = {}
    for label, mask in [("twos", shots.is_three == 0), ("threes", shots.is_three == 1)]:
        sub = shots[mask]
        _, t2 = sh.fit_prior(sh.summarize(sub, ["season", "player_id"]), min_fga=75)
        by_type[label] = {"tau_pts_per_shot": round(t2**0.5, 4),
                          "stabilization_fga": round(sh.stabilization_point(sub, t2))}
    metrics["shrinkage"] = {
        "prior_mean_pts_per_shot": round(mu, 4),
        "prior_sd_pts_per_shot": round(tau2**0.5, 4),
        "stabilization_fga": round(stab),
        "implied_reliability_at_median": round(
            split["median_fga"] / (split["median_fga"] + stab), 3),
        "split_half": {k: round(v, 3) for k, v in split.items()},
        "by_shot_type": by_type,
    }
    plots.reliability_curve(stab, split, DOCS / "reliability.png")

    # 4. Backtest: does shrinkage predict next season better than the raw number?
    pairs = analysis.year_over_year(ps)
    bt = pd.concat([analysis.backtest(pairs, mu, m) for m in (50, 100, 250)])
    metrics["backtest"] = bt.round(4).to_dict(orient="records")
    plots.backtest_bars(bt, DOCS / "backtest.png")
    print(bt.to_string(index=False))

    # 5. Teams: what is stable - shot quality or shot-making?
    metrics["half_season_stability"] = {
        "offense": analysis.half_season_stability(shots, "team"),
        "defense": analysis.half_season_stability(shots, "opponent"),
    }
    cur = shots[shots.season == TEST_SEASON]
    teams = analysis.team_table(cur)
    teams.round(4).to_csv(ART / f"teams_{TEST_SEASON}.csv", index=False)
    plots.team_quadrant(teams, "off", team, DOCS / "team_offense.png", label_offset=(28, 34))
    plots.team_quadrant(teams, "def", team, DOCS / "team_defense.png")

    # 6. Team case study.
    t = cur[cur.team == team]
    metrics["case_study"] = {"team": team, **teams.set_index("team").loc[team].round(4).to_dict()}
    zones = analysis.zone_profile(cur, "team")
    league_zone = analysis.zone_profile(cur).assign(team="NBA")
    zp = pd.concat([zones[zones.team == team], league_zone])
    zp.round(4).to_csv(ART / f"zone_profile_{team}_{TEST_SEASON}.csv", index=False)
    rim_rank = {}
    for season, s in shots.groupby("season"):
        z = analysis.zone_profile(s, "team")
        rim = z[z.zone == "Rim"].set_index("team")["share"]
        rim_rank[season] = {"rim_share": round(rim[team], 4),
                            "rank_of_30": int(rim.rank(ascending=False)[team]),
                            "league_median": round(rim.median(), 4)}
    metrics["case_study"]["rim_share_by_season"] = rim_rank
    metrics["case_study"]["reallocation_5pct_to_rim"] = analysis.reallocate(
        zones[zones.team == team],
        {("Mid-range", "Rim"): 0.025, ("Paint (non-RA)", "Rim"): 0.025})
    metrics["case_study"]["fga_per_game"] = round(len(t) / t["game_id"].nunique(), 1)
    plots.shot_diet_map(t, cur, f"{team} shot diet vs league, {TEST_SEASON}",
                        DOCS / f"shot_diet_{team}.png")
    roster = sh.shrink(sh.summarize(t, ["player_id", "player"]), mu, tau2)
    roster = roster[roster.fga >= 150]
    roster.round(4).to_csv(ART / f"players_{team}_{TEST_SEASON}.csv", index=False)
    plots.player_intervals(roster, f"{team} shot-making, {TEST_SEASON} (≥150 FGA)",
                           DOCS / f"players_{team}.png")

    # Outputs for the app and for reuse.
    ps.round(5).to_csv(ART / "player_seasons.csv", index=False)
    cols = ["season", "game_id", "player_id", "player", "team", "opponent", "zone_basic",
            "action_family", "x_ft", "y_ft", "dist_ft", "is_three", "shot_value", "made", "xfg"]
    shots[cols].assign(xfg=shots["xfg"].round(4)).to_parquet(ART / "shots_xfg.parquet")
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps({k: metrics[k] for k in ("shrinkage", "half_season_stability",
                                               "case_study")}, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="PHX")
    main(parser.parse_args().team)
