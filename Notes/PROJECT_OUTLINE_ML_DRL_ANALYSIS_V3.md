# Modular ML-DRL Analysis & Display Project — Outline Version 3 (Production-Ready with Backtesting & Robustness)

This outline is an evolved synthesis of the A. Belantari references, Code_Skills implementations, and production finance best practices. It extends **V2** with **backtesting frameworks** (walk-forward validation, rolling windows), **regime analysis**, **error handling**, **transaction costs**, and **comparative strategy evaluation**—ensuring the system works reliably in real-world trading environments.

---

## 1. High-Level Architecture (Enhanced for Production)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED INTERFACE (CLI / Config)                            │
│  run_pipeline(config)  →  data  →  model  →  backtest  →  analysis  →  display │
└──────────────────────────────────────────────────────────────────────────────┘
         │                  │           │           │             │             │
         ▼                  ▼           ▼           ▼             ▼             ▼
   ┌──────────┐      ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ config/  │      │ data/    │ │ models/  │ │backtest/ │ │analysis/ │ │ display/ │
   │ pipeline │      │ load     │ │ ml_dl    │ │ portfolio│ │ metrics  │ │matplotlib│
   │ .yaml    │      │ features │ │ drl      │ │ trading  │ │ risk     │ │ plots    │
   │validation│      │ sequence │ │ optim    │ │ regime   │ │ summary  │ │ tables   │
   └──────────┘      │ validate │ │ ensemble │ │ validate │ │ compare  │ │ export   │
                     └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

**Key Enhancements in V3:**
- **Data Validation**: Checks for missing data, outliers, data leakage, stationarity.
- **Backtesting Layer**: Walk-forward validation, rolling windows, transaction costs, slippage.
- **Regime Analysis**: Market regime detection (bull/bear/sideways), regime-aware metrics.
- **Error Handling**: Comprehensive try-catch with logging, API fallbacks, graceful degradation.
- **Ensemble Models**: Combine multiple forecasting/DRL agents for robustness.
- **Comparative Display**: Side-by-side strategy performance, sensitivity analysis.

---

## 2. Data Layer — Production-Grade Pipeline with Validation

### 2.1 Data Ingestion & Validation Module

**New:** `src/data/validator.py` — Pre-flight checks before pipeline execution.

```python
class DataValidator:
    - check_missing_data(df, threshold=0.05): Fail if >5% NaNs
    - check_outliers(df, method='iqr'|'zscore'): Flag/remove extremes
    - check_stationarity(series): ADF test for unit roots
    - check_data_leakage(X_train, X_test): Ensure no temporal overlap
    - check_collinearity(df, threshold=0.9): Warn on multicollinearity
```

**Entry Point:** `validate_data(config)` called before any modeling.

### 2.2 Enhanced Liquidity & Volatility Filtering (Composite Scoring)

From Code_Skills pt4, implement **three-tier filtering**:

1. **Liquidity Tier**: Amihud illiquidity ratio ranked; exclude if trading cost too high.
2. **Volatility Tier**: Annualized volatility threshold; exclude if too high/low.
3. **Composite Tier**: Combined score (normalized vol + normalized illiquidity), top-N tickers.

**Output:** Filtered ticker list with scores for display/reporting.

### 2.3 Unified Data Pipeline with Error Recovery

**Pipeline Class:** `src/data/pipeline.py — DataPipeline(config)`

**Steps (with error handling at each stage):**
1. Download OHLCV (retry logic for API failures, fallback to cached data).
2. Clean & align (fill gaps, validate OHLCV constraints).
3. Add technical indicators (RSI, MACD, ATR, Bollinger, etc., from `ohlcv_indicators.py`).
4. **NEW**: Validate data (check NaNs, outliers, stationarity).
5. Apply liquidity/volatility filtering and composite scoring.
6. Feature selection (correlation + PCA).
7. **NEW**: Check data leakage before split.
8. Scale (RevinTransform or MinMaxScaler).
9. Build sequences or NeuralForecast format.

**Output**: `DataBundle` with train/test sets, metadata, validation results, and error logs.

### 2.4 Data API with Logging & Metrics

```python
def get_data(config, validate=True, log_path=None):
    """
    Input: Config with all parameters.
    Output: DataBundle with X_train, y_train, X_test, y_test, metadata, validation_report.
    Logs all steps to log_path if provided.
    """
    bundle = DataBundle()
    logger = setup_logging(log_path)
    
    try:
        # Download, clean, validate, etc.
    except DataValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise
    except APIError as e:
        logger.warning(f"API error, using cached data: {e}")
        # fallback logic
    
    return bundle
```

---

## 3. Model Layer — Registry with Ensemble & Error Handling

### 3.1 Base Model Interface with Error Handling

**`src/models/base.py`:**

```python
class BaseForecaster:
    def build(self, input_shape, config): pass
    def fit(self, X_train, y_train, X_val, y_val, config):
        """Returns (model, history) with error handling."""
    def predict(self, X_test):
        """Returns predictions; raises PredictionError if model not fitted."""
    def get_hyperparams(self): """Return fitted hyperparameters for logging."""

class BaseDRLAgent:
    def build(self, state_dim, action_dim, config): pass
    def train(self, env, config):
        """Returns (agent, training_history) with early stopping."""
    def act(self, state): """Deterministic action."""
    def act_stochastic(self, state): """Exploratory action."""
    def evaluate(self, env, episodes=10):
        """Compute test performance without learning."""
```

### 3.2 Forecasting Registry (NeuralForecast + Custom PyTorch)

**Primary**: NeuralForecast (LSTM, NHITS, TFT).

**Custom PyTorch** (with hyperparameter tuning from Code_Skills):
- **LSTM**: Basic + L1/L2 regularization, batch norm, dropout.
- **LSTM-CNN**: Conv1d → MaxPool → LSTM.
- **LSTM-Attention**: Multi-head attention over time axis.
- **LSTM-GRU Hybrid**: Bidirectional LSTM/GRU stacking.

**Hyperparameter Tuning** (from deep learning file pt1):
- Grid search or random search over: neurons, dropout, learning_rate, batch_size, epochs.
- Early stopping on validation loss + model checkpoint.
- Correlation-based feature selection before training to reduce noise.

**Error Handling:**
```python
def fit_with_tuning(X_train, y_train, X_val, y_val, config):
    """Grid search + early stopping + best model save."""
    best_model, best_history, best_params = None, None, None
    for hyperparams in config['hyperparameter_grid']:
        try:
            model, history = build_and_fit(hyperparams, X_train, y_train, X_val, y_val)
            val_loss = min(history['val_loss'])
            if val_loss < best_val_loss:
                best_model, best_history, best_params = model, history, hyperparams
        except Exception as e:
            logger.warning(f"Hyperparams {hyperparams} failed: {e}")
            continue
    return best_model, best_history, best_params
```

### 3.3 DRL Registry (PPO, A2C, DDPG, SAC, TD3)

**Environment:** Custom `TradingEnv` (Gym-compatible).

**State Space**: `[indicators_t, portfolio_value_t, holdings_t, cash_t]`.

**Action Space**: Discrete (hold/buy/sell per asset) or continuous (weight allocations).

**Reward Function** (Sharpe-based or returns-based, with regularization).

**Stability Features**:
- Gradient clipping, learning rate annealing.
- Action clipping (avoid extreme trades).
- Episode truncation (max steps = N trading days).

### 3.4 Portfolio Optimization Registry (Expanded)

**Comparison Set** (from Code_Skills):
1. **Risk Parity**: Inverse volatility weighting.
2. **Inverse Volatility**: Simple weighted average.
3. **ERC**: Equal Risk Contribution (scipy minimize).
4. **RB**: Risk Budgeting (predefined budgets).
5. **MDP**: Most Diversified Portfolio.
6. **MVO**: Mean-Variance Optimization.
7. **HRP**: Hierarchical Risk Parity.
8. **HCBAA**: Hierarchical Clustering Based Allocation.
9. **Omega**: Maximize Omega ratio.
10. **CVaR**: Minimize CVaR (95%/99%).
11. **FGP**: Shannon entropy (if implemented).
12. **Efficient Frontier**: Pypfopt with multiple target returns.

**Output:** Standardized `PortfolioWeights` object with strategy name, weights dict, and metadata.

---

## 4. Backtesting Layer (NEW in V3) — Walk-Forward & Rolling Window

**`src/backtest/`** — Production-grade backtesting.

### 4.1 Walk-Forward Validation Framework

**Concept**: For each test period, retrain model and rebalance portfolio.

```python
class WalkForwardBacktester:
    def __init__(self, data, config):
        """data: full OHLCV, config: train_window, test_window, rebalance_freq."""
    
    def run(self, model_builder, optimizer_builder):
        """
        Loop over time windows:
        1. [t0, t0+train_window): Train model & optimizer.
        2. [t0+train_window, t0+train_window+test_window): Test & trade.
        3. Record PnL, weights, predictions.
        4. Slide window forward by rebalance_freq.
        
        Returns: Backtest report with cumulative returns, metrics, regime info.
        """
```

### 4.2 Transaction Costs & Slippage

**Model realistic trading:**
```python
class PortfolioSimulator:
    def rebalance(self, old_weights, new_weights, prices, bid_ask_spread=0.001, commission=0.001):
        """
        Calculate turnover, costs, effective returns.
        turnover = sum(|new_weights - old_weights|) / 2
        cost = turnover * (bid_ask_spread + commission)
        """
```

### 4.3 Regime Detection & Analysis

**From Code_Skills pt2/pt3**, detect market regimes using:
- **Volatility regimes**: Low/medium/high based on rolling volatility percentiles.
- **Return regimes**: Bull (positive returns), bear (negative), sideways (low vol + near-zero returns).
- **Correlation regimes**: High/low asset correlation.

**Regime-Aware Metrics:**
```python
def compute_regime_metrics(returns, regime_labels):
    """
    For each regime:
    - Sharpe ratio
    - Sortino ratio
    - Max drawdown
    - Win rate
    - Return volatility
    
    Returns: DataFrame with regime breakdown.
    """
```

### 4.4 Performance Attribution

```python
class PerformanceAttribution:
    def factor_contribution(self, returns, weights, factor_returns):
        """Barra-style attribution: return = asset_selection + allocation_timing."""
    
    def timing_analysis(self, weights, returns):
        """Analyze if portfolio was overweighted in high-return periods."""
```

---

## 5. Analysis Layer — Enhanced Metrics & Comparative Analysis

### 5.1 Forecasting Evaluation (with Directional & Probabilistic Metrics)

```python
def compute_forecasting_metrics(actual, predicted, ticker, model_name):
    """
    Point metrics: MSE, MAPE, MAE.
    Directional: Up/Down hit rate, false positive rate.
    Probabilistic: CRPS (Continuous Ranked Probability Score) if probabilistic forecast.
    
    Output: DataFrame with all metrics.
    """
```

### 5.2 Portfolio Risk Metrics (Comprehensive via QuantStats)

**Core Metrics** (via quantstats):
- Sharpe, Omega, Sortino, Calmar, Information Ratio.
- CVaR (95%, 99%), Max Drawdown, Drawdown Duration.
- Skewness, Kurtosis, Value at Risk (VaR).
- Hit Rate (% positive days), Win/Loss ratio.

**Regime-Conditioned Metrics**:
- Sharpe by regime (bull/bear/sideways).
- Max drawdown by regime.
- Performance correlation with market regime.

### 5.3 Comparative Strategy Analysis

```python
def compare_strategies(results_dict):
    """
    Input: {strategy_name: strategy_results, ...}
    
    Outputs:
    1. Summary table: Strategy name, annual return, volatility, Sharpe, max DD, etc.
    2. Rolling metrics: Rolling Sharpe, rolling volatility over time.
    3. Drawdown comparison: Max DD, average DD, DD duration.
    4. Regime performance: Metric breakdowns by market regime.
    5. Correlation: How correlated are strategy returns?
    6. Sensitivity: How do results change with ±10% in key parameters?
    
    Returns: Structured dict of DataFrames for display.
    """
```

### 5.4 Model Validation & Diagnostics

```python
def validate_model(model, X_test, y_test):
    """
    Residual analysis: Mean, std, autocorrelation of residuals.
    Calibration: Are prediction intervals correct width?
    Out-of-sample stability: Does performance degrade over time?
    
    Returns: Diagnostic report (warnings if issues detected).
    """
```

---

## 6. Display Layer — Comparative & Regime-Aware Visualizations

### 6.1 Backtesting Visualizations

1. **Equity curve**: Cumulative returns over time, with regime backgrounds (colors).
2. **Drawdown chart**: Running max / current value, highlight max DD.
3. **Rolling metrics**: Rolling Sharpe, volatility, correlation over time.
4. **Rebalancing events**: Marks on equity curve where weights changed.
5. **Turnover over time**: Trading activity per rebalance period.

### 6.2 Strategy Comparison Dashboard

1. **Summary metrics table**: All strategies side-by-side.
2. **Cumulative returns**: Multi-strategy overlay.
3. **Rolling Sharpe**: Multi-strategy rolling metric.
4. **Regime performance heatmap**: Strategies × Regimes, color-coded metric (e.g., Sharpe).
5. **Correlation matrix**: Strategy returns correlation.

### 6.3 Model Diagnostics Plot

1. **Residuals**: Time series, histogram, Q-Q plot.
2. **Predictions vs actuals**: Scatter + regression line.
3. **Prediction errors over time**: Any degradation?
4. **Correlation heatmap**: Features + target (rank by importance).

### 6.4 Enhanced Display API

```python
# Backtesting plots
plot_equity_curve_with_regimes(returns, regime_labels, strategies)
plot_drawdown_analysis(returns, strategies)
plot_rolling_metrics(returns, window=60, metrics=['sharpe', 'volatility', 'correlation'])
plot_rebalancing_activity(dates, turnover, weights)

# Comparative analysis
plot_strategy_comparison_dashboard(results_dict)
plot_regime_performance_heatmap(regime_metrics)
plot_sensitivity_analysis(parameter, metric_range)

# Model diagnostics
plot_residuals_diagnostic(residuals)
plot_predictions_vs_actuals(actual, predicted, ticker, model)
plot_feature_importance(model, feature_names, top_n=15)

# Export
export_backtest_report(results_dict, path)  # PDF/HTML report
export_csv_metrics(results_dict, path)
```

---

## 7. Unified Interface with Experiment Orchestration

### 7.1 Config Schema (Expanded)

```yaml
# src/data configuration
data:
  tickers: [...]
  period: "2y"
  time_step: 10
  look_ahead: 1
  liquidity_filter: {method: "amihud", threshold: 0.001}
  volatility_filter: {min: 0.1, max: 0.5}
  feature_selection: {method: "correlation", threshold: 0.5}
  validation:
    check_stationarity: true
    check_leakage: true
    outlier_method: "iqr"

# Backtesting configuration
backtest:
  train_window: 252  # days
  test_window: 63
  rebalance_freq: 21
  transaction_cost: 0.001  # 0.1%
  bid_ask_spread: 0.001
  regime_detection: {method: "volatility", periods: 63}

# Model configuration (supports multiple models for ensemble)
models:
  - type: "LSTM"
    hyperparameters:
      neurons: [32, 64, 128]
      dropout: [0.2, 0.3]
      learning_rate: [0.001, 0.0005]
    tuning: {method: "grid", early_stopping: true}
  - type: "NHITS"
    hyperparameters: {...}

# Optimization methods
optimization:
  methods: [risk_parity, omega, cvar, hrp, efficient_frontier]
  params:
    omega: {target_return: 0.0}
    cvar: {confidence: 0.95}

# Analysis & display
analysis:
  metrics: [forecasting, risk, regime, comparative]
  regime_conditional: true
  sensitivity_analysis: {parameters: [look_ahead, train_window], range: [±10%]}

display:
  plots: [equity_curve, rolling_sharpe, regime_heatmap, diagnostics]
  export_format: [pdf, html, csv]
  run_id: "exp_20260129_001"
```

### 7.2 Pipeline Execution with Logging & Checkpointing

```python
def run_pipeline(config_path, output_dir):
    """
    1. Load & validate config.
    2. Setup logging to output_dir/logs/.
    3. Get data (with validation).
    4. Run walk-forward backtest:
       a. For each window:
          - Train models (with hyperparameter tuning).
          - Rebalance portfolio (each method).
          - Simulate trading (with costs).
          - Record metrics & regime info.
    5. Analyze (comparative, regime-aware).
    6. Display & export.
    
    Checkpoint after each major step (data, models, backtest) for resumability.
    """
```

### 7.3 Entry Points

```bash
# Single run
python -m src.run_pipeline --config configs/default.yaml --output outputs/exp_001

# Experiment (grid search over configs)
python -m src.run_experiment --config_dir configs/ --experiment exp_001 --output outputs/

# Validation only
python -m src.validate_data --config configs/default.yaml

# Resume from checkpoint
python -m src.run_pipeline --config configs/default.yaml --resume outputs/exp_001/checkpoint.pkl
```

---

## 8. Directory Layout (V3 Additions)

```
project_root/
├── configs/
│   ├── default.yaml
│   ├── dl_portfolio.yaml
│   ├── drl_ppo.yaml
│   └── experiment_grid.yaml  # NEW: multiple configs
├── src/
│   ├── data/
│   │   ├── validator.py      # NEW: DataValidator class
│   │   ├── pipeline.py       # Enhanced with error handling
│   │   ├── sequence.py
│   │   └── api.py
│   ├── backtest/             # NEW: Backtesting framework
│   │   ├── walker.py         # WalkForwardBacktester
│   │   ├── simulator.py      # PortfolioSimulator with costs
│   │   ├── regime.py         # Regime detection & analysis
│   │   └── attribution.py    # Performance attribution
│   ├── models/
│   │   ├── base.py           # Enhanced with error handling
│   │   ├── tuning.py         # NEW: Hyperparameter tuning
│   │   ├── ensemble.py       # NEW: Ensemble methods
│   │   └── ... (rest as before)
│   ├── analysis/
│   │   ├── forecasting.py
│   │   ├── risk.py
│   │   ├── summary.py
│   │   ├── selection.py
│   │   ├── comparative.py    # NEW: Strategy comparison
│   │   └── diagnostics.py    # NEW: Model validation
│   ├── display/
│   │   ├── plots.py          # Enhanced with backtest/regime plots
│   │   ├── tables.py
│   │   ├── export.py         # NEW: PDF/HTML/CSV export
│   │   └── style.py
│   ├── utils/
│   │   ├── logging.py        # NEW: Centralized logging
│   │   ├── checkpoint.py     # NEW: Save/load checkpoints
│   │   └── errors.py         # NEW: Custom exceptions
│   └── run_pipeline.py       # Main orchestrator (enhanced)
├── requirements.txt
├── tests/
│   ├── test_validator.py     # NEW
│   ├── test_backtest.py      # NEW
│   ├── test_regime.py        # NEW
│   └── ... (rest as before)
└── README.md
```

---

## 9. Error Handling & Robustness Strategy (NEW in V3)

### 9.1 Custom Exceptions

```python
# src/utils/errors.py
class DataValidationError(Exception): pass
class APIError(Exception): pass
class PredictionError(Exception): pass
class BacktestError(Exception): pass
class ConfigError(Exception): pass
```

### 9.2 Logging & Monitoring

```python
# src/utils/logging.py
def setup_logging(log_dir, run_id):
    """
    Create logger that logs to:
    1. Console (INFO level).
    2. File (DEBUG level) at log_dir/run_id.log.
    3. Error file at log_dir/run_id_errors.log.
    """
```

### 9.3 Graceful Degradation

- If API fails: Use cached data with warning.
- If hyperparameter tuning fails for a config: Skip and log, continue with defaults.
- If regime detection fails: Use uniform regime labels.
- If a model doesn't converge: Return last checkpoint with diagnostic warning.

### 9.4 Checkpointing

```python
# src/utils/checkpoint.py
def save_checkpoint(state_dict, path):
    """Save pipeline state (data, models, backtest results) for resumption."""

def load_checkpoint(path):
    """Resume pipeline from checkpoint."""
```

---

## 10. Key Improvements in V3 (Summary)

| Feature | V2 | V3 |
|---------|----|----|
| **Validation** | Minimal | Comprehensive (stationarity, leakage, outliers) |
| **Backtesting** | Train/test split only | Walk-forward, rolling windows, transaction costs |
| **Regime Analysis** | None | Bull/bear/sideways detection, regime-aware metrics |
| **Error Handling** | Limited | Comprehensive try-catch, logging, graceful degradation |
| **Hyperparameter Tuning** | Not systematic | Grid/random search with early stopping |
| **Model Comparison** | Separate runs | Unified comparative framework |
| **Ensemble Models** | None | Support for combining multiple forecasters/DRL agents |
| **Performance Attribution** | Basic metrics | Barra-style attribution, timing analysis |
| **Sensitivity Analysis** | None | Parameter sweep analysis |
| **Reporting** | CSV/PNG | PDF/HTML/CSV with automated formatting |
| **Checkpointing** | None | Save/resume capability |

---

## 11. Implementation Quality & Feasibility

### Quality Assessment
**V3 is production-ready in design**, addressing:
- **Robustness**: Comprehensive error handling and logging ensure reliable execution.
- **Realism**: Walk-forward validation, transaction costs, and regime analysis reflect real-world trading.
- **Reproducibility**: Config-driven, checkpointing, and detailed logging ensure experiments are repeatable.
- **Maintainability**: Modular design, clear interfaces, and comprehensive tests make future enhancements easy.

### Ability to Implement
- **Feasibility**: 65-75% ready with existing code. Backtesting (15%), regime analysis (10%), error handling (10%) are new.
- **Timeline**: 6-8 months for full V3 implementation (solo developer).
- **Effort Distribution**:
  - Data validation & error handling: 2 weeks.
  - Backtesting framework: 3-4 weeks.
  - Regime analysis: 2 weeks.
  - Comparative analysis & display: 2-3 weeks.
  - Testing & documentation: 2-3 weeks.

### Critical Success Factors
1. **Start with walk-forward backtest** (core new feature).
2. **Leverage Code_Skills examples** for hyperparameter tuning and regime detection.
3. **Modular error handling** from day one (prevents downstream cascades).
4. **Early validation testing** (unit tests for validator, backtest, regime modules).

---

## 12. Validation & Testing Strategy (NEW in V3)

### Unit Tests
```
tests/test_validator.py:         Test data validation (stationarity, leakage, outliers).
tests/test_backtest.py:          Test walk-forward logic, transaction costs.
tests/test_regime.py:            Test regime detection & conditional metrics.
tests/test_tuning.py:            Test hyperparameter grid/random search.
tests/test_comparative.py:        Test strategy comparison metrics.
```

### Integration Tests
- Full pipeline on small synthetic dataset.
- Walk-forward backtest with known results.
- Regime detection on historical regimes.

### Regression Tests
- V2 outputs should still be reproducible (backward compatibility).
- Existing model interfaces should not break.

---

## Conclusion: V3 as a Production Framework

**V3 transforms the research-focused V2 into a production trading system** by:
1. **Validating data rigorously** to prevent garbage-in-garbage-out.
2. **Backtesting realistically** with walk-forward validation and transaction costs.
3. **Detecting market regimes** to understand when strategies work/fail.
4. **Comparing strategies comprehensively** across regimes and metrics.
5. **Handling errors gracefully** with logging and checkpointing for reliability.

The outline is implementable with disciplined, incremental development. Start with the backtesting framework (the most critical new piece), then add regime analysis, error handling, and reporting. This ensures core functionality works before embellishments.
