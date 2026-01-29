# Feature selection (correlation, PCA) (V3)

from typing import Any, List, Optional

import numpy as np
import pandas as pd


def select_features_correlation(
    df: pd.DataFrame,
    target: Optional[pd.Series] = None,
    threshold: float = 0.5,
) -> List[str]:
    """
    Select features by correlation with target or drop highly correlated pairs.

    Args:
        df: Feature DataFrame.
        target: Optional target series.
        threshold: Correlation threshold.

    Returns:
        List of selected column names.
    """
    if target is not None:
        corr = df.corrwith(target).abs()
        return corr[corr >= threshold].index.tolist()
    corr = df.corr().abs()
    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    ).stack()
    to_drop = upper[upper >= threshold].index.get_level_values(1).unique()
    return [c for c in df.columns if c not in to_drop]
