import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error calculation"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA scoring function from Saxena et al. 2008 / Arias Chao et al. 2020.

    Asymmetric: over-prediction (predicting more life than there is) is
    penalized exp(d/10); under-prediction exp(d/13). Over-prediction is
    far more costly because it means the engine fails before scheduled
    maintenance: the dollar-weighted real-world loss

    Lower is better. The score grows exponentially with error, so a few
    very wrong predictions dominate the metric: same as in safety-critical
    deployment
    """
    delta = y_pred - y_true  # positive = over-prediction
    score = np.where(
        delta < 0,
        np.exp(-delta / 13.0),  # under-prediction (delta < 0)
        np.exp( delta / 10.0),  # over-prediction (delta >= 0)
    )
    return float(np.sum(score))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute both metrics and log them"""
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "n": len(y_true),
    }
    logger.info("[%s] RMSE=%.3f, NASA=%.1f, n=%d", label, metrics["rmse"], metrics["nasa_score"], metrics["n"])
    return metrics


def per_unit_evaluation(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    """Break out RMSE and NASA-score per unit"""
    rows = []
    for unit, g in df.groupby("unit"):
        rows.append({
            "unit": unit,
            "n_cycles": len(g),
            "rmse": rmse(g[y_true_col].values, g[y_pred_col].values),
            "nasa_score": nasa_score(g[y_true_col].values, g[y_pred_col].values),
        })
    return pd.DataFrame(rows).set_index("unit")