"""Shot Quality Explorer - interactive front end over the pipeline's artifacts.

Run: streamlit run app/streamlit_app.py   (after scripts/run_pipeline.py)
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from shotquality import analysis, plots
from shotquality import shrinkage as sh

ART = Path(__file__).resolve().parents[1] / "artifacts"

st.set_page_config(page_title="Shot Quality Explorer", layout="wide")


@st.cache_data
def load():
    shots = pd.read_parquet(ART / "shots_xfg.parquet")
    metrics = json.loads((ART / "metrics.json").read_text())
    return shots, metrics


shots, metrics = load()
prior = metrics["shrinkage"]
mu, tau2 = prior["prior_mean_pts_per_shot"], prior["prior_sd_pts_per_shot"] ** 2

with st.sidebar:
    st.header("Filters")
    seasons = sorted(shots["season"].unique(), reverse=True)
    season = st.selectbox("Season", seasons)
    teams = sorted(shots["team"].unique())
    team = st.selectbox("Team", teams, index=teams.index("PHX"), key="team")
    st.caption(
        f"Shot-making stabilizes at ~{prior['stabilization_fga']} FGA. "
        "Estimates below that are mostly noise and are shrunk hard toward league average."
    )

cur = shots[shots["season"] == season]
team_shots = cur[cur["team"] == team]

st.title("Shot Quality Explorer")
st.caption(
    "Expected FG% (xFG) from a gradient-boosted model of shot location and context, scored "
    "out-of-season. Shot quality = expected points per shot (xPPS). Shot-making = actual "
    "minus expected points, shrunk with empirical Bayes."
)

tab_team, tab_players, tab_whatif = st.tabs(["Team", "Players", "Shot-diet what-if"])

with tab_team:
    t = analysis.team_table(cur)
    row = t.set_index("team").loc[team]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Offense xPPS", f"{row.off_xpps:.3f}", f"rank {row.off_xpps_rank} of 30",
              delta_color="off")
    c2.metric("Offense shot-making /100", f"{row.off_making * 100:+.1f}",
              f"rank {row.off_making_rank}", delta_color="off")
    c3.metric("Defense xPPS allowed", f"{row.def_xpps:.3f}", f"rank {row.def_xpps_rank} of 30",
              delta_color="off")
    c4.metric("Opp. shot-making /100", f"{row.def_making * 100:+.1f}",
              f"rank {row.def_making_rank}", delta_color="off")

    stab = metrics["half_season_stability"]
    st.info(
        f"What persists? First-half vs second-half correlation across team-seasons: "
        f"defensive shot quality allowed r = {stab['defense']['xpps']:.2f}, "
        f"opponent shot-making r = {stab['defense']['raw']:.2f}. Judge defenses mainly on "
        "the shots they allow."
    )

    for side, label in [("off", "Offense"), ("def", "Defense")]:
        d = t.assign(making=t[f"{side}_making"] * 100, xpps=t[f"{side}_xpps"],
                     highlight=t["team"] == team)
        chart = alt.Chart(d).mark_circle(size=110, stroke="white", strokeWidth=1.5).encode(
            x=alt.X("xpps:Q", scale=alt.Scale(zero=False),
                    title="xPPS" if side == "off" else "xPPS allowed (lower is better)"),
            y=alt.Y("making:Q", title="Shot-making per 100 shots" if side == "off"
                    else "Opponent shot-making per 100 shots"),
            color=alt.condition("datum.highlight", alt.value(plots.ORANGE), alt.value(plots.MUTED)),
            tooltip=["team", alt.Tooltip("xpps:Q", format=".3f"),
                     alt.Tooltip("making:Q", format="+.1f")],
        )
        labels = chart.mark_text(dx=14, fontSize=11).encode(text="team")
        st.subheader(label)
        st.altair_chart((chart + labels).properties(height=380), width="stretch")

    zp = pd.concat([analysis.zone_profile(team_shots).assign(team=team),
                    analysis.zone_profile(cur).assign(team="NBA")])
    st.subheader("Shot diet by zone")
    st.dataframe(
        zp.pivot_table(index="zone", columns="team", values=["share", "xpps", "pps"],
                       observed=True).round(3),
        width="stretch",
    )
    buf = io.BytesIO()
    plots.shot_diet_map(team_shots, cur, f"{team} shot diet vs league, {season}", buf)
    st.image(buf.getvalue(), width=640)

with tab_players:
    min_fga = st.slider("Minimum FGA", 50, 800, 150, step=25)
    ps = sh.shrink(sh.summarize(team_shots, ["player_id", "player"]), mu, tau2)
    ps = ps[ps["fga"] >= min_fga].sort_values("eb", ascending=False)
    view = ps.assign(
        **{"Shot-making (raw)": ps["raw"] * 100, "Shot-making (shrunk)": ps["eb"] * 100,
           "90% low": ps["eb_lo"] * 100, "90% high": ps["eb_hi"] * 100,
           "Signal share": ps["reliability"]}
    )[["player", "fga", "fg_pct", "xfg_pct", "xpps", "Shot-making (raw)", "Shot-making (shrunk)",
       "90% low", "90% high", "Signal share"]]
    st.dataframe(view.round(3), width="stretch", hide_index=True)

    if len(ps):
        who = st.selectbox("Shot chart", ps["player"])
        p = team_shots[team_shots["player"] == who]
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(p["x_ft"], p["y_ft"], c=p["xfg"], cmap=plots.DIVERGING.reversed(),
                        vmin=0.25, vmax=0.75, s=14, alpha=0.8, linewidths=0)
        plots.draw_court(ax)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label("xFG of the attempt")
        cb.outline.set_visible(False)
        ax.set_title(f"{who}, {season}: {len(p)} attempts colored by shot quality")
        st.pyplot(fig)

with tab_whatif:
    st.write(
        "Move a share of attempts between zones and price the change at this team's own "
        "expected points per shot in each zone. This values shot *selection* only: it assumes "
        "the added attempts are as good as the team's current ones in that zone."
    )
    prof = analysis.zone_profile(team_shots)
    zones = analysis.ZONE_ORDER
    c1, c2, c3 = st.columns(3)
    src = c1.selectbox("From", zones, index=zones.index("Mid-range"))
    dst = c2.selectbox("To", zones, index=zones.index("Rim"))
    max_share = float(prof.set_index("zone").loc[src, "share"]) * 100
    amount = c3.slider("Percent of all attempts", 0.0, round(max_share, 1),
                       min(5.0, round(max_share, 1)), step=0.5)
    if src != dst:
        out = analysis.reallocate(prof, {(src, dst): amount / 100})
        fga_pg = len(team_shots) / team_shots["game_id"].nunique()
        m1, m2, m3 = st.columns(3)
        m1.metric("xPPS now", f"{out['xpps_before']:.3f}")
        m2.metric("xPPS after", f"{out['xpps_after']:.3f}", f"{out['delta_per_100']:+.2f} / 100")
        m3.metric("Points per game", f"{out['delta_per_100'] * fga_pg / 100:+.2f}",
                  f"at {fga_pg:.0f} FGA per game", delta_color="off")
    st.dataframe(prof[["zone", "fga", "share", "fg", "xfg", "pps", "xpps"]].round(3),
                 hide_index=True, width="stretch")
