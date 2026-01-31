Summary:

This document maps the implementation to the code spec in Notes\PATCHTST_IMPLEMENTATION_OUTLINE.md and Notes\PROJECT_OUTLINE_ML_DRL_ANALYSIS_V3.md.

1) Data acquisition & preprocessing (Phase 1)
- Implemented in src\data\patchtst_pipeline.py (PatchTSTDataPipeline).
- Downloads OHLCV via src\ohlcv\yfinance_ohlcv.py for symbols in src\symbols\portfolio (configurable in configs\forecast_patchtst.yaml).
- Cleaning: duplicate removal, sort by time, forward/back fill, IQR-based clipping for outliers.
- Normalization: StandardScaler fit on training split only; applied to validation/test in run\run_pipeline_forecast_patchTST.py.

2) Feature engineering (Phase 1.2)
- Implemented in src\data\patchtst_pipeline.py (FeatureEngineer).
- Momentum indicators: RSI(14/21/28), Stoch RSI %K/%D, MACD line/signal/hist, ROC(12/24), Momentum(10/20).
- Trend indicators: SMA(20/50/100/200), EMA(12/26), ADX(14), linear regression slope(20/50), TEMA(14).
- Volatility indicators: Bollinger Bands (upper/lower/%B), ATR(14), Std(20), Coef of Variation.
- Volume indicators: Volume ROC, OBV, CMF, Volume SMA ratio, VWAP.
- Price action: returns (1/5/10/20), log returns, high-low ratio, close-open ratio, true range, upper/lower shadow ratios.
- Statistical: rolling skew/kurtosis (20/50), autocorrelation lags 1-5, cross-correlation with volume.

3) Target variable creation (Phase 1.3)
- Implemented in src\data\patchtst_pipeline.py.
- Targets created for price (close) and returns; multi-step horizons are built by create_multistep_sequences in the runner.

4) Feature selection & correlation analysis (Phase 2) — ENHANCED
- select_features_patchtst() in src\analysis\selection.py: correlation threshold, high-correlation pair removal (|r|>0.95), VIF (max 5), top-N cap (default 40).
- VIF skipped when >80 features to avoid numerical issues; fallback if VIF drops all features.
- Correlation heatmap and feature-vs-target bar plot in src\display\forecast_plots.py.

5) Model architecture (Phase 3)
- PatchTST model in src\models\patchtst.py: patch embedding, positional encoding, encoder stack, forecast head.
- Configurable via configs\forecast_patchtst.yaml → PatchTSTConfig.

6) Data preparation for training (Phase 4)
- Time-series split (split_time_series) and sequence generation (create_multistep_sequences) in src\data\patchtst_pipeline.py.
- DataLoaders in run\run_pipeline_forecast_patchTST.py.
- Auto-adjustment for limited data: seq_len, pred_len, patch_len, stride clamped when needed.

7) Training loop (Phase 5)
- PatchTSTTrainer in src\models\patchtst.py: Adam, ReduceLROnPlateau, MSE, gradient clipping, early stopping.
- Training history CSV and PNG plots saved.

8) Prediction intervals (Phase 6.3)
- Parametric 80%/95% intervals from validation residual std (z = 1.2816 / 1.96).

9) Metrics & evaluation (Phase 7) — ENHANCED
- Point metrics: MSE, RMSE, MAE, MAPE (zeros excluded), SMAPE, R², MASE.
- MASE: MAE(forecast)/MAE(naive); MASE < 1 = better than baseline.
- Directional accuracy, interval coverage, horizon metrics, Information Ratio.
- Benchmark alignment fixed: reindex to test dates before computing IR.

10) Visualization suite (Phase 8.2) — ENHANCED
- Plots: forecast vs actual, prediction intervals, error histograms, horizon metrics, calibration curve, correlation heatmap, feature-correlation bar.
- New: training history curves, scatter actual vs predicted (price and returns).

11) Data validation (V3 — PROJECT_OUTLINE Section 2.1)
- DataValidator integrated before modeling: check_missing_data, check_stationarity (returns), check_collinearity.

12) Outputs & artifacts (Phase 8.3)
- Metrics CSV, models (patchtst_price.pt, patchtst_returns.pt), plots, logs.

13) Pipeline entrypoint
- run\run_pipeline_forecast_patchTST.py — reads configs\forecast_patchtst.yaml.

---

Verification (run 2026-01-31):
- Pipeline runs end-to-end without errors.
- Feature selection: 795 → 40 features (corr, pair removal, top-N).
- MASE computed; PatchTST returns MASE < 1 (outperforms naive).
- Training and scatter plots generated.
- Limited data: seq_len/pred_len auto-reduced (e.g. 504→10, 30→10) when insufficient samples.