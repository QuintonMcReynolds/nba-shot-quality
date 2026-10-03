import numpy as np
import pandas as pd

from shotquality import data, model


def test_action_family_mapping():
    assert data.action_family("Driving Floating Jump Shot") == "floater"
    assert data.action_family("Pullup Jump shot") == "pull_up"
    assert data.action_family("Running Alley Oop Dunk Shot") == "dunk"
    assert data.action_family("Jump Shot") == "jump_shot"
    assert data.action_family("Something New") == "other"


def test_team_abbreviations_resolved_from_games():
    df = pd.DataFrame({"GAME_ID": [1, 1, 2, 2], "TEAM_ID": [10, 20, 10, 30],
                       "HTM": ["PHX", "PHX", "DEN", "DEN"], "VTM": ["LAL", "LAL", "PHX", "PHX"]})
    assert data.team_abbreviations(df) == {10: "PHX", 20: "LAL", 30: "DEN"}


def test_season_label():
    assert data.season_label(2025) == "2025-26"
    assert data.season_label(1999) == "1999-00"


def _shots(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-25, 25, n), rng.uniform(-4, 30, n)
    dist = np.hypot(x, y)
    p = np.clip(0.7 - 0.012 * dist, 0.2, 0.8)
    return pd.DataFrame({
        "x_ft": x, "y_ft": y, "dist_ft": dist, "angle_deg": np.degrees(np.arctan2(abs(x), y)),
        "is_three": (dist > 23.75).astype(int), "period": rng.integers(1, 5, n),
        "secs_left_period": rng.integers(0, 720, n), "end_of_period": 0, "home": 1,
        "action_family": rng.choice(["jump_shot", "layup", "dunk"], n),
        "zone_basic": np.where(dist < 4, "RA", "other"), "zone_area": "C", "zone_range": "x",
        "made": (rng.random(n) < p).astype(int), "true_p": p,
    })


def test_xfg_model_learns_distance_and_is_calibrated():
    train, test = _shots(seed=1), _shots(5000, seed=2)
    m = model.XFGModel(model.FEATURE_SETS["location+context"]).fit(train)
    p = m.predict_proba(test)
    assert np.corrcoef(p, test["true_p"])[0, 1] > 0.9
    assert abs(p.mean() - test["made"].mean()) < 0.02


def test_xfg_model_handles_unseen_category():
    train, test = _shots(seed=1), _shots(100, seed=2)
    m = model.XFGModel(model.FEATURE_SETS["location+context"]).fit(train)
    test["action_family"] = "never_seen"
    assert np.isfinite(m.predict_proba(test)).all()


def test_zone_baseline_returns_zone_rates():
    df = pd.DataFrame({"zone_basic": ["a", "a", "b"], "zone_area": "c", "zone_range": "r",
                       "made": [1, 0, 1]})
    p = model.ZoneBaseline().fit(df).predict_proba(df)
    assert np.allclose(p, [0.5, 0.5, 1.0])


def test_calibration_table_covers_all_shots():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, 5000)
    y = (rng.random(5000) < p).astype(int)
    cal = model.calibration_table(y, p, bins=10)
    assert cal["n"].sum() == 5000
    assert np.allclose(cal["pred"], cal["actual"], atol=0.06)
