"""Plots for PatchTST forecasting."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def plot_forecast_vs_actual(
    actual: pd.Series,
    predicted: pd.Series,
    title: str,
    ax=None,
):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    ax.plot(actual.index, actual.values, label="actual")
    ax.plot(predicted.index, predicted.values, label="predicted")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_prediction_intervals(
    actual: pd.Series,
    predicted: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    title: str,
    ax=None,
):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    ax.plot(actual.index, actual.values, label="actual")
    ax.plot(predicted.index, predicted.values, label="predicted")
    ax.fill_between(actual.index, lower.values, upper.values, alpha=0.2, label="interval")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_error_distribution(errors: np.ndarray, title: str, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    ax.hist(errors, bins=min(50, max(10, len(errors) // 5)), density=True)
    ax.set_title(title)
    return ax


def plot_horizon_metrics(metrics: Dict[str, np.ndarray], title: str, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    for name, values in metrics.items():
        ax.plot(np.arange(1, len(values) + 1), values, label=name)
    ax.set_xlabel("horizon")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_calibration_curve(nominal: Iterable[float], empirical: Iterable[float], title: str, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    nominal = np.array(list(nominal))
    empirical = np.array(list(empirical))
    ax.plot(nominal, empirical, marker="o", label="empirical")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    ax.set_xlabel("nominal")
    ax.set_ylabel("empirical")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_correlation_heatmap(corr: pd.DataFrame, title: str, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax)
    return ax


def plot_feature_correlation_bar(corr: pd.Series, title: str, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(8, 4))
    corr = corr.sort_values(ascending=False).head(20)
    ax.bar(corr.index, corr.values)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90)
    return ax