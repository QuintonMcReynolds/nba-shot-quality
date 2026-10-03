"""Expected field-goal percentage (xFG) models and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

LOCATION_FEATURES = ["x_ft", "y_ft", "dist_ft", "angle_deg", "is_three"]
CONTEXT_FEATURES = ["action_family", "period", "secs_left_period", "end_of_period", "home"]
FEATURE_SETS = {
    "location": LOCATION_FEATURES,
    "location+context": LOCATION_FEATURES + CONTEXT_FEATURES,
}
CATEGORICAL = {"action_family"}
ZONE_KEYS = ["zone_basic", "zone_area", "zone_range"]


class ZoneBaseline:
    """League FG% by NBA shot-chart zone - the 'traditional' shot-quality benchmark."""

    def fit(self, df: pd.DataFrame) -> ZoneBaseline:
        self.rates_ = df.groupby(ZONE_KEYS)["made"].mean()
        self.overall_ = df["made"].mean()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(df[ZONE_KEYS])
        return self.rates_.reindex(idx).fillna(self.overall_).to_numpy()


class XFGModel:
    """Gradient-boosted make probability from shot location (+ optional context)."""

    def __init__(self, features: list[str], random_state: int = 0):
        self.features = features
        self.clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=600,
            max_leaf_nodes=31,
            min_samples_leaf=200,
            l2_regularization=1.0,
            categorical_features=[f in CATEGORICAL for f in features],
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=random_state,
        )

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for col in CATEGORICAL & set(self.features):
            X[col] = X[col].astype("category")
        return X

    def fit(self, df: pd.DataFrame) -> XFGModel:
        X = self._X(df)
        for col in CATEGORICAL & set(self.features):
            self.categories_ = {col: X[col].cat.categories}
        self.clf.fit(X, df["made"])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._X(df)
        for col, cats in getattr(self, "categories_", {}).items():
            values = X[col].astype(object)
            X[col] = pd.Categorical(values.where(values.isin(cats)), categories=cats)
        return self.clf.predict_proba(X)[:, 1]


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "mean_pred": float(p.mean()),
        "mean_actual": float(np.mean(y)),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 20) -> pd.DataFrame:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    t = pd.DataFrame({"bin": idx, "p": p, "y": y}).groupby("bin").agg(
        pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    t["se"] = np.sqrt(t["actual"] * (1 - t["actual"]) / t["n"])
    return t.reset_index(drop=True)


def out_of_season_predictions(shots: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Leave-one-season-out xFG, so no player-season is scored by a model that saw it."""
    out = np.empty(len(shots))
    for season in shots["season"].unique():
        test = (shots["season"] == season).to_numpy()
        model = XFGModel(features).fit(shots.loc[~test])
        out[test] = model.predict_proba(shots.loc[test])
    return out
