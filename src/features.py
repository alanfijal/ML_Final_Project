import logging
from dataclasses import dataclass, field 
from pathlib import Path 
import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

W_VARS = ["alt", "Mach", "TRA", "T2"] #input var 

INTERIM_DIR = Path(__file__).resolve().parents[1] / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class RegimeClusterer:
    """
    Sample to an operating regime (climb -> cruise -> descent -> idle)
    Uses k-means++ initialisation with 10 random restarts
    """
    n_regimes: int = 6
    random_state: int = 42
    _scaler: StandardScaler | None = field(default=None, init=False)
    _kmeans: KMeans | None = field(default=None, init=False)

    def fit(self, df: pd.DataFrame, sample_size: int = 500_000) -> "RegimeClusterer":
        logger.info("Fitting RegimeClusterer with k=%d on up to %d samples", self.n_regimes, sample_size)
        n = min(sample_size, len(df))
        idx = np.random.default_rng(self.random_state).choice(len(df), size=n, replace=False)
        X = df.iloc[idx][W_VARS].to_numpy()

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)

        self._kmeans = KMeans(
            n_clusters=self.n_regimes,
            init="k-means++",
            n_init=10,
            random_state=self.random_state
        ).fit(Xs)

        logger.info("Cluster inertia: %.0f", self._kmeans.inertia_)
        return self 
    

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._kmeans is None or self._scaler is None:
            raise RuntimeError("RegimeClusterer must be fit before predict.")
        
        X = self._scaler.transform(df[W_VARS].to_numpy())
        return self._kmeans.predict(X)
    

    def describe_regimes(self, df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Return mean {alt, Mach, TRA, T2} per regime, used as naming util"""
        tmp = df[W_VARS].copy()
        tmp["regime"] = labels
        summary = tmp.groupby("regime").agg(["mean", "size"])
        return summary
    

def elbow_study(
        df: pd.DataFrame,
        k_values: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
        sample_size: int = 200_000,
        random_state: int = 42
) -> pd.DataFrame:
    """
    Fit k-means for a range of k and return inertia. 
    Used for elbow method of selecting optimal k === K where the marginal reduction in inertia stabilises
    """
    logger.info("Elbow study: k=%s, sample_size=%d", k_values, sample_size)
    n = min(sample_size, len(df))
    idx = np.random.default_rng(random_state).choice(len(df), size=n, replace=False)
    X = df.iloc[idx][W_VARS].to_numpy()
    Xs = StandardScaler().fit_transform(X)

    rows = []
    for k in k_values:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=random_state
        ).fit(Xs)

        rows.append({"k": k, "inertia": km.inertia_})
        logger.info("  k=%d -> inertia=%.0f", k, km.inertia_)
    
    return pd.DataFrame(rows)



def aggregate_to_cycle_level(
        df: pd.DataFrame,
        sensor_cols: list[str],
        regime_labels: np.ndarray | None = None,
        n_regimes: int | None = None
) -> pd.DataFrame:
    """
    Aggregate 1HZ data to one row per (unit, cycle)
    Features per cycle:
      - For each sensor: global mean, std, min, max,
        and (if regime_labels given) per-regime mean.
      - For each W variable: global mean and std.
      - Regime mix: fraction of samples spent in each regime.
      - Flight duration in seconds (= number of 1Hz samples).
      - Cycle index (raw cycle number — useful baseline feature).
      - RUL label (constant within a cyclemax

    Returns:
        df indexed by (unit, cycle), one row per flight
    """
    logger.info("Aggregating %d rows to cycle level...", len(df))
    if regime_labels is not None:
        df = df.assign(_regime=regime_labels)
    
    #Global aggregates
    agg_funcs = ["mean", "std", "min", "max"]
    agg_dict = {c: agg_funcs for c in sensor_cols + W_VARS}
    global_agg = df.groupby(["unit", "cycle"]).agg(agg_dict)
    global_agg.columns = [f"{c}_{stat}" for c, stat in global_agg.columns]
    logger.debug("Global aggregates: %d features", global_agg.shape[1])

    #Per-regime mean of each sensor 
    if regime_labels is not None:
        assert n_regimes is not None
        regime_means = (
            df.groupby(["unit", "cycle", "_regime"])[sensor_cols]
              .mean()
              .unstack("_regime")
        )
        regime_means.columns = [
            f"{sensor}_r{int(r)}" for sensor, r in regime_means.columns
        ]

        for sensor in sensor_cols:
            global_col = f"{sensor}_mean"
            for r in range(n_regimes):
                col = f"{sensor}_r{r}"
                if col in regime_means.columns:
                    regime_means[col] = regime_means[col].fillna(
                        global_agg[global_col]
                    )
        logger.debug("Regime-conditioned features: %d", regime_means.shape[1])
    
    else:
        regime_means = None 

    #Regime mix per cycle
    if regime_labels is not None:
        regime_counts = (
            df.groupby(["unit", "cycle", "_regime"])
            .size()
            .unstack("_regime", fill_value=0)
        )
        regime_mix = regime_counts.div(regime_counts.sum(axis=1), axis=0)
        regime_mix.columns = [f"regime_mix_r{int(r)}" for r in regime_mix.columns]
    
    else:
        regime_mix = None 

    #Flight duration and cycle index
    duration = (
        df.groupby(["unit", "cycle"])
          .size()
          .rename("flight_duration_s")
          .to_frame()
    )
    cycle_index = (
        df.groupby(["unit", "cycle"])
          .first()[[]]
          .reset_index()
          .set_index(["unit", "cycle"])
    )
    cycle_index["cycle_idx"] = cycle_index.index.get_level_values("cycle")

    #RUL label
    rul = df.groupby(["unit", "cycle"])["RUL"].first().to_frame()

    #Blocks combined
    blocks = [global_agg, duration, cycle_index, rul]
    if regime_means is not None:
        blocks.append(regime_means)
    if regime_mix is not None:
        blocks.append(regime_mix)

    out = pd.concat(blocks, axis=1)
    logger.info("Cycle-level features: %d rows × %d columns",
                out.shape[0], out.shape[1])
    return out

def piecewise_linear_rul(rul: pd.Series, r_max: float = 50.0) -> pd.Series:
    """Early-life RUL isn't predictable from sensors (no degradation has happened yet),
    so we treat any RUL > r_max as 'plenty of life left'"""
    return rul.clip(upper=r_max)


def save_parquet(df: pd.DataFrame, name: str) -> Path:
    path = INTERIM_DIR / f"{name}.parquet"
    df.to_parquet(path)
    logger.info("Wrote %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def load_parquet(name: str) -> pd.DataFrame:
    path = INTERIM_DIR / f"{name}.parquet"
    logger.info("Reading %s", path)
    return pd.read_parquet(path)