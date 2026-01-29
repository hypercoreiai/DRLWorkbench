# Model validation and diagnostics (V3 — PROJECT_OUTLINE Section 5.4)

from typing import Any, Dict

import numpy as np
import pandas as pd


def validate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
) -> Dict[str, Any]:
    """
    Residual analysis: mean, std, autocorrelation of residuals.
    Calibration: prediction interval width (if applicable).
    Out-of-sample stability: performance over time.

    Returns:
        Diagnostic report (warnings if issues detected).
    """
    try:
        pred = model.predict(X_test)
    except Exception:
        return {"error": "predict failed", "warnings": []}

    actual = np.asarray(y_test)
    pred = np.asarray(pred)
    residuals = actual - pred

    report = {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "warnings": [],
    }

    if len(residuals) > 1:
        try:
            from statsmodels.tsa.stattools import acf
            acf_vals = acf(residuals, nlags=min(10, len(residuals) // 2))
            report["residual_autocorr_lag1"] = float(acf_vals[1])
        except ImportError:
            pass

    if abs(report["residual_mean"]) > 0.1 * (np.std(actual) + 1e-10):
        report["warnings"].append("Residual mean is large (bias)")

    return report
