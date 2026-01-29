# Ensemble methods: combine multiple forecasters/DRL agents (V3 — PROJECT_OUTLINE Section 3.4)

from typing import Any, Dict, List, Optional

import numpy as np


def ensemble_predict(
    models: List[Any],
    X: Any,
    method: str = "mean",
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Combine predictions from multiple models.

    Args:
        models: List of fitted models with .predict(X).
        X: Input features.
        method: 'mean', 'median', or 'weighted'.
        weights: Weights per model (for 'weighted').

    Returns:
        Combined prediction array.
    """
    preds = [m.predict(X) for m in models]
    stack = np.array(preds)
    if method == "mean":
        return np.nanmean(stack, axis=0)
    if method == "median":
        return np.nanmedian(stack, axis=0)
    if method == "weighted" and weights is not None:
        w = np.array(weights)
        w = w / w.sum()
        return np.average(stack, axis=0, weights=w)
    return np.nanmean(stack, axis=0)
