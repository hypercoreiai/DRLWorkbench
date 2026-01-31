# Feature selection (correlation, VIF, PCA) (V3 - PATCHTST Phase 2)

from typing import Any, List, Optional, Tuple

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


def compute_vif(df: pd.DataFrame) -> pd.Series:
    """
    Compute Variance Inflation Factor for each feature.
    VIF > 5 indicates multicollinearity; > 10 is severe.

    Args:
        df: Numeric DataFrame (no NaNs).

    Returns:
        Series of VIF values indexed by column name.
    """
    from sklearn.linear_model import LinearRegression

    X = df.fillna(0).values
    n_cols = X.shape[1]
    vif = np.zeros(n_cols)
    for i in range(n_cols):
        y = X[:, i]
        X_i = np.delete(X, i, axis=1)
        reg = LinearRegression().fit(X_i, y)
        r2 = reg.score(X_i, y)
        vif[i] = 1.0 / (1.0 - r2) if r2 < 1 else np.inf
    return pd.Series(vif, index=df.columns)


def select_features_patchtst(
    df: pd.DataFrame,
    target: pd.Series,
    min_corr: float = 0.05,
    max_pair_corr: float = 0.95,
    max_vif: float = 5.0,
    top_n: Optional[int] = 40,
) -> Tuple[List[str], dict]:
    """
    PatchTST-compliant feature selection (Phase 2 outline):
    - Drop features with |corr(target)| < min_corr
    - Drop one from each highly correlated pair (|r| > max_pair_corr)
    - Drop features with VIF > max_vif
    - Keep top_n by correlation strength

    Args:
        df: Feature DataFrame.
        target: Target series (aligned with df).
        min_corr: Min |correlation| with target to keep.
        max_pair_corr: Max |correlation| between feature pairs.
        max_vif: Max allowed VIF.
        top_n: Max features to keep (None = no cap).

    Returns:
        (selected_columns, report_dict)
    """
    common = df.index.intersection(target.index)
    df_a = df.loc[common].dropna(axis=1, how="all")
    target_a = target.loc[common].dropna()
    common = df_a.index.intersection(target_a.index)
    df_a = df_a.loc[common]
    target_a = target_a.loc[common]

    report = {"initial": len(df_a.columns), "after_corr": 0, "after_pair": 0, "after_vif": 0, "final": 0}

    # 1) Correlation with target
    corr = df_a.corrwith(target_a).abs()
    sel = corr[corr >= min_corr].index.tolist()
    report["after_corr"] = len(sel)
    if not sel:
        return list(df.columns[: min(top_n or 999, len(df.columns))]), report

    df_sel = df_a[sel].dropna(axis=1, how="all")
    sel = list(df_sel.columns)

    # 2) Remove highly correlated pairs
    corr_mat = df_sel.corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool)).stack()
    to_drop = set()
    for (a, b), v in upper.items():
        if v >= max_pair_corr:
            # Drop the one with lower correlation to target
            to_drop.add(a if corr.get(a, 0) <= corr.get(b, 0) else b)
    sel = [c for c in sel if c not in to_drop]
    report["after_pair"] = len(sel)
    if not sel:
        return list(df.columns[: min(top_n or 999, len(df.columns))]), report

    df_sel = df_a[sel].fillna(0)
    if len(df_sel.columns) < 2:
        report["after_vif"] = len(sel)
        report["final"] = len(sel)
        return sel, report

    # 3) VIF (skip if too many features; fallback if VIF drops all)
    sel_before_vif = sel.copy()
    if len(sel) <= 80:
        try:
            vif = compute_vif(df_sel)
            sel_vif = [c for c in sel if vif.get(c, np.inf) <= max_vif]
            if sel_vif:
                sel = sel_vif
        except Exception:
            pass
    if len(sel) == 0:
        sel = sel_before_vif
    report["after_vif"] = len(sel)

    # 4) Top-N by correlation
    corr_sel = corr.reindex(sel).dropna().sort_values(ascending=False)
    if len(corr_sel) > 0:
        if top_n is not None and len(corr_sel) > top_n:
            sel = corr_sel.head(top_n).index.tolist()
        else:
            sel = corr_sel.index.tolist()
    report["final"] = len(sel)
    if not sel:
        sel = corr.abs().sort_values(ascending=False).head(top_n or 40).index.tolist()
        report["final"] = len(sel)
    return sel, report
