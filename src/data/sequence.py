# Sequence building for forecasting / NeuralForecast format (V3)

from typing import Any, Dict, Optional

import numpy as np


def build_sequences(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    time_step: int = 10,
    look_ahead: int = 1,
) -> tuple:
    """
    Build sliding-window sequences for time series models.

    Args:
        X: Features (T, F).
        y: Target (T,) or None.
        time_step: Input sequence length.
        look_ahead: Forecast horizon.

    Returns:
        (X_seq, y_seq) or (X_seq,) if y is None.
        X_seq shape: (samples, time_step, F).
    """
    T = len(X)
    if T < time_step + look_ahead:
        return (np.empty((0, time_step, X.shape[1])), np.empty(0))
    X_seq = np.array(
        [X[i : i + time_step] for i in range(T - time_step - look_ahead + 1)]
    )
    if y is not None:
        y_seq = np.array(
            [y[i + time_step + look_ahead - 1] for i in range(len(X_seq))]
        )
        return X_seq, y_seq
    return (X_seq,)
