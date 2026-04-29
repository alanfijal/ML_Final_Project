import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _subsample(df: pd.DataFrame, n_per_unit: int = 50_000,seed: int = 42) -> pd.DataFrame:
    """Stratified by unit so every engine is fairly represented."""
    rng = np.random.default_rng(seed)
    parts = []
    for u, g in df.groupby("unit"):
        take = min(n_per_unit, len(g))
        idx = rng.choice(len(g), size=take, replace=False)
        parts.append(g.iloc[idx])
    sub = pd.concat(parts, ignore_index=True)
    logger.debug("Subsampled %d -> %d rows for plotting", len(df), len(sub))
    return sub


def plot_flight_envelope_kde(dev: pd.DataFrame, test: pd.DataFrame, fname: str = "fig03_flight_envelope_kde.png",) -> Path:
    """Reproduces Figure 3 of the Arias Chao et al. paper: per-unit KDE of altitude, Mach, TRA, and T2. Test units 14 and 15 should visibly divergefrom the training distribution."""
    logger.info("Plotting flight envelope KDE...")

    combined = pd.concat([
        dev.assign(role="train"),
        test.assign(role="test"),
    ], ignore_index=True)
    sub = _subsample(combined, n_per_unit=50_000)

    variables = [
        ("alt",  "Altitude [ft]"),
        ("Mach", "Mach Number [-]"),
        ("TRA",  "Throttle Resolver Angle [%]"),
        ("T2",   "Fan inlet temperature T2 [°R]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (col, label) in zip(axes.flat, variables):
        for unit, g in sub.groupby("unit"):
            role = g["role"].iloc[0]
            sns.kdeplot(
                g[col], ax=ax,
                label=f"Unit {unit} ({role})",
                linewidth=1.5,
                linestyle="-" if role == "train" else "--",
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Density")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flat:
        ax.get_legend().remove() if ax.get_legend() else None
    fig.legend(handles, labels, loc="center right", fontsize=8,
               bbox_to_anchor=(1.12, 0.5))
    fig.suptitle("Flight envelope distributions per unit", fontsize=13)
    fig.tight_layout()

    path = FIG_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_rul_trajectories(dev: pd.DataFrame, test: pd.DataFrame, fname: str = "rul_trajectories.png") -> Path:
    """RUL vs cycle for every unit. Confirms each engine runs to failure"""
    logger.info("Plotting RUL trajectories...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, df, title in [(axes[0], dev, "Train units"),
                          (axes[1], test, "Test units")]:
        per_cycle = df.groupby(["unit", "cycle"])["RUL"].first().reset_index()
        for unit, g in per_cycle.groupby("unit"):
            ax.plot(g["cycle"], g["RUL"], label=f"Unit {unit}", linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Flight cycle")
        ax.set_ylabel("RUL [cycles]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = FIG_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_single_flight(df: pd.DataFrame, unit: int, cycle: int, fname: str = "single_flight_trace.png",) -> Path:
    """One flight, four scenario descriptors. Reproduces Figure 4: shows that 
    each cycle contains climb / cruise / descent — motivates cycle-level
    aggregation rather than treating 1 Hz samples as i.i.d."""
    logger.info("Plotting single flight trace for unit=%d, cycle=%d", unit, cycle)
    flight = df[(df["unit"] == unit) & (df["cycle"] == cycle)].reset_index(drop=True)
    if flight.empty:
        raise ValueError(f"No data for unit={unit}, cycle={cycle}")

    variables = [("alt", "Altitude [ft]"),
                 ("Mach", "Mach [-]"),
                 ("TRA", "TRA [%]"),
                 ("T2", "T2 [°R]")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for ax, (col, label) in zip(axes.flat, variables):
        ax.plot(flight.index, flight[col], linewidth=0.8, color="#8B0000")
        ax.set_xlabel("Time within flight [s]")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Single flight trace — Unit {unit}, Cycle {cycle}", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path