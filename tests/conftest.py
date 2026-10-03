import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_shots():
    """Shots from 400 players with known shot-making skill, on a league-like difficulty mix."""
    rng = np.random.default_rng(7)
    n_players, per_player = 400, 600
    skill = rng.normal(0, 0.03, n_players)  # true make-probability boost
    rows = []
    for pid in range(n_players):
        dist = rng.uniform(0, 28, per_player)
        is_three = (dist > 23.75).astype(int)
        xfg = np.clip(0.68 - 0.012 * dist, 0.25, 0.75)
        made = rng.random(per_player) < np.clip(xfg + skill[pid], 0.01, 0.99)
        rows.append(pd.DataFrame({
            "player_id": pid, "game_id": rng.integers(0, 82, per_player), "dist_ft": dist,
            "is_three": is_three, "shot_value": 2 + is_three, "xfg": xfg, "made": made.astype(int),
        }))
    return pd.concat(rows, ignore_index=True), skill
