import pandas as pd
import pytest

from shotquality import analysis


@pytest.fixture
def profile():
    return pd.DataFrame({"zone": ["Rim", "Mid-range", "Above-break 3"],
                         "share": [0.3, 0.3, 0.4], "xpps": [1.3, 0.8, 1.05]})


def test_reallocate_no_moves_is_neutral(profile):
    out = analysis.reallocate(profile, {})
    assert out["delta_per_100"] == pytest.approx(0)


def test_reallocate_prices_moved_shots(profile):
    out = analysis.reallocate(profile, {("Mid-range", "Rim"): 0.1})
    assert out["delta_per_100"] == pytest.approx(0.1 * (1.3 - 0.8) * 100)


def test_reallocate_rejects_impossible_move(profile):
    with pytest.raises(ValueError):
        analysis.reallocate(profile, {("Mid-range", "Rim"): 0.31})


def test_backtest_shapes():
    pairs = pd.DataFrame({"fga_t": [300, 300, 300], "fga_t1": [300, 300, 300],
                          "raw_t": [0.1, -0.1, 0.0], "eb_t": [0.05, -0.05, 0.0],
                          "raw_t1": [0.04, -0.06, 0.01]})
    bt = analysis.backtest(pairs, 0.0, min_fga_t=100).set_index("estimator")
    assert bt.loc["Shrunk (EB)", "wrmse"] < bt.loc["Raw", "wrmse"]
