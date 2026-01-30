# DRLWorkbench

**Deep Reinforcement Learning Backtesting and Analysis Framework**

A comprehensive Python framework for developing, testing, and deploying Deep Reinforcement Learning agents for quantitative finance applications.

## Features

### ✅ Implemented Core Features

1. **Backtesting Framework** — Walk-forward validation, rolling windows, transaction costs, slippage modeling with realistic portfolio simulation
2. **Regime Analysis** — Detect bull/bear/sideways markets using K-Means, GMM, or rule-based methods; compute regime-conditioned metrics
3. **Error Handling & Logging** — Comprehensive exception hierarchy, centralized logging with rotation, checkpointing for resume capability
4. **Data Validation** — Statistical tests (stationarity via ADF test), data quality checks (outliers, missing data), type validation

### 🚧 Planned Features (See Documentation)

5. **Hyperparameter Tuning** — Systematic grid search + random search with early stopping
6. **Comparative Strategy Analysis** — Side-by-side comparison, sensitivity analysis, performance attribution
7. **Ensemble Models** — Framework for combining multiple forecasters/DRL agents
8. **Enhanced Visualizations** — Equity curves with regime backgrounds, rolling metrics, interactive dashboards
9. **Professional Reporting** — PDF/HTML/CSV export with automated formatting

## Installation

```bash
# Clone the repository
git clone https://github.com/hypercoreiai/DRLWorkbench.git
cd DRLWorkbench

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
import numpy as np
from drlworkbench import setup_logger, DataValidator, RegimeDetector

# Set up logging
logger = setup_logger("my_analysis")

# Generate or load your data
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
}, index=pd.date_range('2020-01-01', periods=252))

# Validate data quality
validator = DataValidator()
results = validator.validate_all(data)

if results['is_valid']:
    logger.info("✓ Data validation passed")
    
    # Detect market regimes
    returns = data['close'].pct_change().dropna()
    detector = RegimeDetector(method='kmeans', n_regimes=3)
    regimes = detector.fit_predict(returns)
    
    logger.info(f"Detected {len(regimes.unique())} regimes")
```

## Documentation

Comprehensive documentation is available in the project:

- **[PROJECT_OUTLINE_V3.md](DRLWorkbench_Notes_MD/PROJECT_OUTLINE_ML_DRL_ANALYSIS_V3.md)** - Detailed project architecture and implementation plan
- **[PROJECT_OUTLINE_V4.md](DRLWorkbench_Notes_MD/PROJECT_OUTLINE_ML_DRL_ANALYSIS_V4.md)** - Enhanced features including MLOps, API service, risk management
- **[PROJECT_QUESTIONS_V3.md](_Notes_MD/PROJECT_QUESTIONS_V3.md)** - 170+ technical questions covering all aspects
- **[PROJECT_ANSWERS_V3.md](_Notes_MD/PROJECT_ANSWERS_V3.md)** - Detailed implementation decisions and rationale

## Project Structure
...
project_root/
├── configs/
│   ├── default.yaml
│   ├── dl_portfolio.yaml
│   └── drl_ppo.yaml
├── src/
│   ├── ohlcv/                    # EXISTING
│   │   ├── _period.py
│   │   ├── yfinance_ohlcv.py
│   │   ├── clean_ohlcv.py
│   │   └── ...
│   ├── indicators/               # EXISTING
│   │   ├── get_indicators.py     # FRED, BEA, yfinance index/stock
│   │   ├── clean_indicators.py
│   │   └── ohlcv_indicators.py   # NEW: Belantari-style tech indicators on OHLCV
│   ├── normalize/                # EXISTING
│   │   ├── revin.py              # RevinTransform + RevIN (PyTorch)
│   │   └── ...
│   ├── data/                     # NEW: unified data API
│   │   ├── __init__.py
│   │   ├── pipeline.py          # DataPipeline(config): OHLCV → clean → indicators → scale → sequences/NF
│   │   ├── sequence.py          # create_dataset, to_neuralforecast_format, volatility helper
│   │   └── api.py               # get_data(config) → X_train, y_train, X_test, y_test, metadata
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # abstract fit/predict or train/act
│   │   ├── neuralforecast/       # NeuralForecast wrappers (LSTM, NHITS, N-BEATS, TFT, etc.)
│   │   │   ├── adapter.py        # fit(train_df), predict() with long-format
│   │   │   └── ...
│   │   ├── pytorch/              # optional custom LSTM, LSTM-CNN, LSTM-Attention (nn.Module)
│   │   ├── drl/                  # PPO, A2C, DDPG (PyTorch)
│   │   │   ├── ppo.py
│   │   │   └── ...
│   │   └── portfolio/            # risk_parity, omega, cvar, mdv, fgp
│   │       ├── risk_parity.py
│   │       ├── omega.py
│   │       └── ...
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── forecasting.py       # MSE, MAPE, MAE
│   │   ├── risk.py              # Sharpe, Omega, CVaR, volatility
│   │   ├── summary.py           # annual return, vol, skew, kurtosis
│   │   └── selection.py         # median / momentum → buy list
│   ├── display/
│   │   ├── __init__.py
│   │   ├── style.py
│   │   ├── plots.py              # loss, actual_vs_pred, heatmap, cumulative_returns
│   │   ├── tables.py
│   │   └── api.py
│   └── run_pipeline.py          # main: config → data → model → analysis → display
├── requirements.txt
└── README.md
...
``` # Configuration files


## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=drlworkbench --cov-report=html

# Run specific test module
pytest tests/test_validation/ -v
```

## Development Status

**Current Version:** 0.1.0 (Alpha)

**Test Coverage:** 76% (22 tests, all passing)

**Security:** No vulnerabilities detected (CodeQL verified)

## Core Modules

### Utils Module
- `setup_logger()` - Centralized logging configuration
- `Checkpoint` - Save/load state for long-running operations
- Custom exception hierarchy for better error handling

### Backtesting Module
- `BacktestEngine` - Core backtesting with transaction costs
- `Portfolio` - Position and cash management
- Performance metrics: Sharpe, Sortino, Calmar, max drawdown

### Regime Detection Module
- `RegimeDetector` - Market state identification
- Methods: K-Means, Gaussian Mixture Model, Rule-based
- Automatic regime labeling (Bear/Sideways/Bull)

### Validation Module
- `DataValidator` - Comprehensive data quality checks
- Statistical tests: ADF (stationarity), outlier detection
- Data type validation and missing data analysis

## Contributing

This project follows professional Python development practices:

- Type hints throughout the codebase
- Comprehensive docstrings (NumPy style)
- Unit tests with pytest
- Code formatting with black
- Linting with flake8

## License

MIT License - see LICENSE file for details

## Authors

DRLWorkbench Team

## Acknowledgments

Built on industry-standard libraries:
- NumPy, Pandas, SciPy for numerical computing
- scikit-learn for machine learning algorithms
- statsmodels for statistical tests
- matplotlib, seaborn, plotly for visualization
