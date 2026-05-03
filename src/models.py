import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.evaluation import rmse, nasa_score

logger = logging.getLogger(__name__)


NON_FEATURE_COLS = {"unit", "cycle", "RUL", "RUL_pw", "cycle_idx"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns in the cycle-level dataframe"""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def leave_one_engine_out_cv(df: pd.DataFrame, target: str = "RUL_pw", model_factory=None, use_scaler: bool = False) -> pd.DataFrame:
    """Run leave-one-engine-out cross-validation

    For each training unit u in the dev set:
      - Train on all other units
      - Predict on unit u
      - Record metrics

    Args:
    df: Cycle-level features for the 6 dev units. Must contain 'unit'
    target: Column to predict. 'RUL_pw' (piecewise-linear) by default
    model_factory: Returns a fresh sklearn-compatible regressor each call. Required
    use_scaler: bool If True, fit a StandardScaler on training features and apply to validation. Needed for Ridge; not for Random Forest

    Returns:
        DataFrame with one row per held-out unit

    """
    if model_factory is None:
        raise ValueError("model_factory is required")

    feats = feature_cols(df)
    logger.info("LOEO CV: %d features, target=%s", len(feats), target)

    rows = []
    for held_out in sorted(df["unit"].unique()):
        train = df[df["unit"] != held_out]
        val = df[df["unit"] == held_out]

        X_tr = train[feats].values
        y_tr = train[target].values
        X_va = val[feats].values
        y_va = val[target].values

        if use_scaler:
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_va = scaler.transform(X_va)

        if np.isnan(X_tr).any() or np.isnan(X_va).any():
            col_means = np.nanmean(X_tr, axis=0)
            X_tr = np.where(np.isnan(X_tr), col_means, X_tr)
            X_va = np.where(np.isnan(X_va), col_means, X_va)

        model = model_factory()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)

        rows.append({
            "held_out_unit": held_out,
            "n_train": len(train),
            "n_val": len(val),
            "rmse": rmse(y_va, y_pred),
            "nasa_score": nasa_score(y_va, y_pred),
        })
        logger.info("  holdout=Unit %d: RMSE=%.3f, NASA=%.1f",
                    held_out, rows[-1]["rmse"], rows[-1]["nasa_score"])

    return pd.DataFrame(rows).set_index("held_out_unit")


@dataclass
class TrainedModel:
    """Bundle a fitted model with its feature list and an optional scaler"""
    model: object
    features: list[str]
    scaler: StandardScaler | None
    target: str

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            X = np.where(np.isnan(X), col_means, X)
        return self.model.predict(X)


def fit_final(df_train: pd.DataFrame, target: str = "RUL_pw", model_factory=None, use_scaler: bool = False) -> TrainedModel:
    """Fit a model on all training data"""
    feats = feature_cols(df_train)
    X = df_train[feats].values
    y = df_train[target].values

    scaler = None
    if use_scaler:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), col_means, X)

    model = model_factory()
    model.fit(X, y)
    logger.info("Final model trained on %d cycles, %d features", len(df_train), len(feats))
    return TrainedModel(model=model, features=feats, scaler=scaler, target=target)



def ridge_factory(alpha: float = 1.0):
    return lambda: Ridge(alpha=alpha, random_state=42)


def rf_factory(n_estimators: int = 300, max_features: str = "sqrt", min_samples_leaf: int = 2, random_state: int = 42):
    return lambda: RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
