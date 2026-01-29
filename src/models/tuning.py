# Hyperparameter tuning: grid/random search, early stopping (V3 — PROJECT_OUTLINE Section 3.2)

from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger("pipeline")


def fit_with_tuning(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    config: Dict[str, Any],
    build_and_fit: Optional[Callable] = None,
) -> Tuple[Optional[Any], Optional[Dict], Optional[Dict]]:
    """
    Grid search + early stopping + best model save.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        config: Must contain 'hyperparameter_grid' or model config.
        build_and_fit: (hyperparams, X_train, y_train, X_val, y_val) -> (model, history).

    Returns:
        (best_model, best_history, best_params) or (None, None, None) if all fail.
    """
    best_model, best_history, best_params = None, None, None
    best_val_loss = float("inf")
    grid = config.get("hyperparameter_grid", [{}])

    for hyperparams in grid:
        try:
            if build_and_fit is None:
                continue
            model, history = build_and_fit(
                hyperparams, X_train, y_train, X_val, y_val
            )
            val_losses = history.get("val_loss", [])
            if not val_losses:
                continue
            val_loss = min(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model, best_history, best_params = (
                    model,
                    history,
                    hyperparams,
                )
        except Exception as e:
            logger.warning("Hyperparams %s failed: %s", hyperparams, e)
            continue

    return best_model, best_history, best_params
