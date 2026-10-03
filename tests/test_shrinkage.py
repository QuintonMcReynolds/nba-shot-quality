import numpy as np

from shotquality import shrinkage as sh


def test_prior_recovers_true_spread(synthetic_shots):
    shots, skill = synthetic_shots
    s = sh.summarize(shots, ["player_id"])
    _, tau2 = sh.fit_prior(s, min_fga=50)
    # Skill is on make probability; residuals are in points, so scale by mean shot value.
    true_sd = skill.std() * shots["shot_value"].mean()
    assert abs(np.sqrt(tau2) - true_sd) / true_sd < 0.2


def test_shrinkage_beats_raw_estimate(synthetic_shots):
    shots, skill = synthetic_shots
    s = sh.summarize(shots, ["player_id"])
    mu, tau2 = sh.fit_prior(s)
    s = sh.shrink(s, mu, tau2)
    truth = skill[s["player_id"]] * shots["shot_value"].mean()
    assert np.mean((s["eb"] - truth) ** 2) < np.mean((s["raw"] - truth) ** 2)
    assert s["reliability"].between(0, 1).all()
    assert (s["eb_lo"] <= s["eb"]).all() and (s["eb"] <= s["eb_hi"]).all()


def test_summarize_points_accounting():
    import pandas as pd

    shots = pd.DataFrame({"g": [1, 1, 1], "made": [1, 0, 1], "xfg": [0.5, 0.5, 0.4],
                          "shot_value": [2, 3, 3]})
    s = sh.summarize(shots, ["g"]).iloc[0]
    assert s["pts"] == 5
    assert np.isclose(s["xpts"], 1.0 + 1.5 + 1.2)
    assert np.isclose(s["raw"], (5 - 3.7) / 3)


def test_stabilization_matches_reliability_definition(synthetic_shots):
    shots, _ = synthetic_shots
    tau2 = 0.004
    n = sh.stabilization_point(shots, tau2)
    per_shot_var = np.mean(shots["shot_value"] ** 2 * shots["xfg"] * (1 - shots["xfg"]))
    assert np.isclose(tau2 / (tau2 + per_shot_var / n), 0.5)


def test_split_half_reliability_detects_signal(synthetic_shots):
    shots, _ = synthetic_shots
    out = sh.split_half_reliability(shots, ["player_id"], min_fga=100)
    assert out["groups"] == 400
    assert 0.2 < out["split_half_r"] < 0.9
