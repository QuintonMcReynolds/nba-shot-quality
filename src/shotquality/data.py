"""Download, load, and clean NBA shot-chart data.

Source: the public mirror of stats.nba.com ShotChartDetail maintained at
https://github.com/shufinskiy/nba_data (one file per regular season; the file's
year is the year the season starts, so 2025 -> 2025-26).
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
URL = "https://github.com/shufinskiy/nba_data/raw/main/datasets/shotdetail_{year}.tar.xz"
DEFAULT_YEARS = (2022, 2023, 2024, 2025)

# ACTION_TYPE has ~50 labels; the long tail is folded into a handful of families.
ACTION_FAMILIES = {
    "Dunk": "dunk",
    "Alley Oop": "dunk",
    "Tip": "putback",
    "Putback": "putback",
    "Hook": "hook",
    "Layup": "layup",
    "Finger Roll": "layup",
    "Floating": "floater",
    "Step Back": "step_back",
    "Pullup": "pull_up",
    "Pull-Up": "pull_up",
    "Fadeaway": "fadeaway",
    "Turnaround": "fadeaway",
    "Running": "running_jumper",
    "Driving": "driving_jumper",
    "Jump Shot": "jump_shot",
}


def season_label(year: int) -> str:
    return f"{year}-{(year + 1) % 100:02d}"


def download(years=DEFAULT_YEARS, raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        csv = raw_dir / f"shotdetail_{year}.csv"
        if csv.exists():
            continue
        archive = raw_dir / f"shotdetail_{year}.tar.xz"
        print(f"downloading {season_label(year)} ...")
        urllib.request.urlretrieve(URL.format(year=year), archive)
        with tarfile.open(archive) as tar:
            tar.extractall(raw_dir, filter="data")


def action_family(action_type: str) -> str:
    # Order matters: "Driving Floating Jump Shot" is a floater, "Running Pull-Up" a pull-up.
    for key, family in ACTION_FAMILIES.items():
        if key.lower() in action_type.lower():
            return family
    return "other"


def team_abbreviations(df: pd.DataFrame) -> dict[int, str]:
    """Map TEAM_ID -> abbreviation: the only abbreviation present in every one of its games."""
    games = df.drop_duplicates(["GAME_ID", "TEAM_ID"])[["TEAM_ID", "HTM", "VTM"]]
    candidates = {
        team_id: set.intersection(*({h, v} for h, v in zip(g.HTM, g.VTM)))
        for team_id, g in games.groupby("TEAM_ID")
    }
    # A team seen in few games can have two candidates; drop ones other teams have claimed.
    out: dict[int, str] = {}
    while len(out) < len(candidates):
        resolved = {t: next(iter(c)) for t, c in candidates.items() if len(c) == 1 and t not in out}
        if not resolved:
            raise ValueError(f"could not resolve team abbreviations: {candidates}")
        out.update(resolved)
        claimed = set(out.values())
        for t, c in candidates.items():
            if t not in out:
                c -= claimed
    return out


def clean(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Add season, team/opponent, and geometric/clock features to raw ShotChartDetail rows."""
    df = df.copy()
    df["season"] = season_label(year)
    abbr = team_abbreviations(df)
    df["team"] = df["TEAM_ID"].map(abbr)
    df["home"] = (df["team"] == df["HTM"]).astype(int)
    df["opponent"] = np.where(df["home"] == 1, df["VTM"], df["HTM"])

    # LOC_X/LOC_Y are tenths of a foot with the hoop at the origin.
    df["x_ft"] = df["LOC_X"] / 10.0
    df["y_ft"] = df["LOC_Y"] / 10.0
    df["dist_ft"] = np.hypot(df["x_ft"], df["y_ft"])
    df["angle_deg"] = np.degrees(np.arctan2(df["x_ft"].abs(), df["y_ft"]))
    df["is_three"] = (df["SHOT_TYPE"] == "3PT Field Goal").astype(int)
    df["shot_value"] = 2 + df["is_three"]
    df["secs_left_period"] = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]
    df["end_of_period"] = (df["secs_left_period"] <= 2).astype(int)
    df["action_family"] = df["ACTION_TYPE"].map(action_family)
    df["made"] = df["SHOT_MADE_FLAG"].astype(int)
    df["game_date"] = pd.to_datetime(df["GAME_DATE"].astype(str), format="%Y%m%d")

    keep = [
        "season", "GAME_ID", "GAME_EVENT_ID", "game_date", "PLAYER_ID", "PLAYER_NAME",
        "team", "opponent", "home", "PERIOD", "secs_left_period", "end_of_period",
        "ACTION_TYPE", "action_family", "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
        "x_ft", "y_ft", "dist_ft", "angle_deg", "is_three", "shot_value", "made",
    ]
    out = df[keep].rename(columns={
        "GAME_ID": "game_id", "GAME_EVENT_ID": "event_id", "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player", "PERIOD": "period", "ACTION_TYPE": "action_type",
        "SHOT_ZONE_BASIC": "zone_basic", "SHOT_ZONE_AREA": "zone_area",
        "SHOT_ZONE_RANGE": "zone_range",
    })
    return out.drop_duplicates(["game_id", "event_id"]).reset_index(drop=True)


def load(years=DEFAULT_YEARS, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    download(years, raw_dir)
    frames = [clean(pd.read_csv(raw_dir / f"shotdetail_{y}.csv"), y) for y in years]
    return pd.concat(frames, ignore_index=True)
