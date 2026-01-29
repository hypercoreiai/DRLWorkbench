# Project Questions — V3 Objective Alignment

Use this list to answer the questions that determine whether the final system meets the **project objective**: a production-ready ML-DRL analysis and display framework that works reliably in real-world trading environments (backtesting, regime analysis, error handling, comparative strategy evaluation).

---

## 1. Project Scope & Success Criteria

| # | Question | Why it matters |
|---|----------|----------------|
| 1.1 | What is the **primary use case**: research/backtesting only, paper trading, or eventual live trading? | Drives how strict validation, costs, and latency need to be. |
| 1.2 | How do you define **“meets the project objective”**? (e.g. minimum Sharpe, max drawdown cap, regime coverage, or “runs end-to-end without manual fixes”.) | Needed to know when the project is “done” and how to prioritize. |
| 1.3 | Who consumes the outputs: only you, a team, or external stakeholders? What format do they need (PDF report, CSV, dashboard, API)? | Shapes display layer and export (PDF/HTML/CSV/dashboard). |
| 1.4 | Is **backward compatibility** with V2 (or existing scripts) required? If yes, which interfaces or outputs must remain unchanged? | Informs regression tests and refactors (Section 12). |

---

## 2. Data & Universe

| # | Question | Why it matters |
|---|----------|----------------|
| 2.1 | What **asset universe**: equities (which indices/exchanges?), crypto (which venues?), FX, or mixed? | Determines data sources (yfinance, Kraken, etc.) and liquidity/vol filters. |
| 2.2 | What **ticker list(s)** and **history depth** (e.g. 2y, 5y) are required for your typical run? | Feeds `configs/*.yaml` (tickers, period) and data pipeline capacity. |
| 2.3 | For **liquidity/volatility filtering** (Section 2.2): Do you need Amihud illiquidity, and what are acceptable min/max volatility and composite top-N? | Drives three-tier filtering implementation and config schema. |
| 2.4 | What **missing-data threshold** is acceptable (e.g. fail if >5% NaNs per series)? Same for **outliers**: flag only, or auto-remove/clip (IQR vs z-score)? | Aligns `DataValidator` and pipeline behavior with your risk tolerance. |
| 2.5 | **Caching**: Where should cached OHLCV/indicators live (local dir, DB), and how long is cache valid before re-download? | Affects API fallback and `get_data()` behavior (Section 2.4). |
| 2.6 | **Feature set**: Fixed list (e.g. RSI, MACD, ATR, Bollinger) or config-driven? Do you need PCA and/or correlation-based feature selection in the pipeline? | Determines scope of `ohlcv_indicators` and feature_selection in config. |

---

## 3. Models & Optimization

| # | Question | Why it matters |
|---|----------|----------------|
| 3.1 | Which **forecasting models** are in scope for V3: only NeuralForecast (LSTM, NHITS, TFT), only custom PyTorch (LSTM, LSTM-CNN, LSTM-Attention, LSTM-GRU), or both? | Sets the “models” registry and tuning surface. |
| 3.2 | Which **DRL agents** are required first: PPO only, or PPO + A2C/DDPG/SAC/TD3? | Determines DRL registry and `TradingEnv` interface. |
| 3.3 | **Action space**: Discrete (hold/buy/sell per asset) or continuous (weight allocations), or both with config switch? | Affects `TradingEnv` and reward design. |
| 3.4 | **Reward**: Sharpe-based, returns-based, or configurable? Any extra terms (e.g. turnover penalty, drawdown penalty)? | Needed for training stability and realism (Section 3.3). |
| 3.5 | Which **portfolio optimization** methods must be implemented for comparison: all 12 from Section 3.4, or a subset (e.g. risk parity, MVO, HRP, CVaR, efficient frontier)? | Drives effort and `PortfolioWeights` output surface. |
| 3.6 | **Ensemble**: Combine forecasters only, DRL only, or both? Aggregation: mean, median, or weighted (and how are weights defined)? | Clarifies `ensemble_predict` and config. |
| 3.7 | **Hyperparameter tuning**: Grid only, or grid + random search? Which params are in the search space (neurons, dropout, lr, batch_size, epochs), and what are acceptable runtimes per run? | Affects `fit_with_tuning` and experiment_grid design. |

---

## 4. Backtesting & Realism

| # | Question | Why it matters |
|---|----------|----------------|
| 4.1 | **Walk-forward**: Default train/test/rebalance windows (e.g. 252/63/21 days)? Fixed or configurable per run? | Core backtest behavior (Section 4.1). |
| 4.2 | **Transaction costs**: Single rate (e.g. 0.1%) or separate commission + bid-ask spread? Same for all assets or per-asset? | Impacts `PortfolioSimulator` and config (Section 4.2). |
| 4.3 | **Slippage**: Ignore, or model (e.g. linear in order size)? If yes, what default assumption? | Realism vs simplicity. |
| 4.4 | **Execution**: Assume fills at close, or do you need intraday/next-open assumptions? | Affects how PnL and turnover are computed. |
| 4.5 | What **backtest output** is mandatory: cumulative returns, Sharpe, max DD, turnover, regime breakdown, and/or full weight history? | Defines `Backtest report` structure and display/export. |

---

## 5. Regime Analysis

| # | Question | Why it matters |
|---|----------|----------------|
| 5.1 | Which **regime types** are required: volatility only, return (bull/bear/sideways), correlation, or all? | Scopes `regime.py` and `regime_detection` in config. |
| 5.2 | **Regime definition**: Rolling window length and percentile bounds (e.g. 63 days, 33/67)? Same for all regime types? | Needed for `detect_regimes()` and reproducibility. |
| 5.3 | Should **strategy evaluation** always be regime-conditioned (Sharpe/DD by regime), or optional? | Drives analysis and display (regime heatmap, Section 6.2). |
| 5.4 | If regime detection **fails** (e.g. too little data): Use uniform regime labels, or fail the run? | Aligns with graceful degradation (Section 9.3). |

---

## 6. Analysis & Reporting

| # | Question | Why it matters |
|---|----------|----------------|
| 6.1 | **Forecasting metrics**: Point (MSE, MAPE, MAE) and directional (hit rate, FP rate) only, or also probabilistic (e.g. CRPS)? | Scopes `compute_forecasting_metrics` (Section 5.1). |
| 6.2 | **Risk metrics**: Use QuantStats for full set (Sharpe, Omega, Sortino, Calmar, CVaR, VaR, etc.) or a custom subset? | Decides dependency and `risk.py` implementation. |
| 6.3 | **Comparative analysis**: Which of the six outputs in `compare_strategies` are required (summary table, rolling metrics, drawdown comparison, regime performance, correlation, sensitivity)? | Prioritizes implementation in analysis and display. |
| 6.4 | **Sensitivity analysis**: Which parameters (e.g. look_ahead, train_window, transaction_cost) and what range (e.g. ±10%)? | Config and comparative module (Section 5.3, 7.1). |
| 6.5 | **Model diagnostics**: Residual analysis and OOS stability only, or also prediction-interval calibration? | Scopes `validate_model` and diagnostics plots (Section 5.4, 6.3). |
| 6.6 | **Export**: PDF and HTML required, or is CSV + PNG sufficient for your workflow? | Drives display/export and dependencies. |

---

## 7. Operational & Deployment

| # | Question | Why it matters |
|---|----------|----------------|
| 7.1 | **Entry points**: Do you need all four (single run, experiment grid, validate_data only, resume from checkpoint), or a subset first? | Prioritizes `run_pipeline`, `run_experiment`, `validate_data` (Section 7.3). |
| 7.2 | **Checkpointing**: What must be resumable (config + data only, or data + models + backtest results)? Any size/time limits? | Defines `save_checkpoint`/`load_checkpoint` payload. |
| 7.3 | **Logging**: Console + file (DEBUG) + error file enough, or do you need structured logs (e.g. JSON) or integration with a monitoring system? | Shapes `setup_logging` (Section 9.2). |
| 7.4 | **Failure policy**: On API failure, use cache and continue; on validation failure, fail fast. Any other cases (e.g. single-model failure in ensemble)? | Aligns with Section 9.3 graceful degradation. |

---

## 8. Prioritization & Timeline

| # | Question | Why it matters |
|---|----------|----------------|
| 8.1 | Given the outline’s **6–8 month** estimate: What is your target timeline, and what can be deferred to a later phase (e.g. DRL, full optimization set, PDF export)? | Drives phasing and “minimum viable V3.” |
| 8.2 | **Critical path**: Agree with “walk-forward backtest first,” then regime, then error handling/reporting? Or different order? | Aligns with Section 11 critical success factors. |
| 8.3 | **Testing**: Unit tests for validator, backtest, regime are in place. Do you need integration tests (full pipeline on synthetic data) and regression tests (V2 compatibility) before calling V3 done? | Matches Section 12 and definition of “production-ready.” |

---

## How to Use This List

1. **Answer each question** (short bullet or 1–2 sentences).
2. **Mark “must have” vs “nice to have”** for scope questions (e.g. 3.5, 6.3, 6.6).
3. **Record answers** in a separate doc (e.g. `PROJECT_ANSWERS_V3.md`) or in the same file under each section.
4. **Review with the outline**: Cross-check Section 7.1 (config), Section 8 (directory layout), and Section 11 (feasibility) so that your answers are implementable.

Once these are fixed, the implementation can be scoped and ordered so that the **final results meet the project objective** as you define it.
