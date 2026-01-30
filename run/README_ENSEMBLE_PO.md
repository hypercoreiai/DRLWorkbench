# Ensemble Portfolio Optimization Pipeline

## Overview

The `run_pipeline_ensemble_po.py` script implements a comprehensive ensemble portfolio optimization pipeline that combines multiple optimization methods with backtesting, regime analysis, and detailed reporting.

## Features

### Portfolio Optimization Methods
1. **Risk Parity** - Inverse volatility weighting for equal risk contribution
2. **Omega Ratio** - Maximize probability-weighted ratio of gains vs losses
3. **CVaR (Conditional Value at Risk)** - Minimize expected loss in worst cases
4. **HRP (Hierarchical Risk Parity)** - Machine learning-based diversification using hierarchical clustering
5. **Efficient Frontier** - Mean-variance optimization (Markowitz)

### Data Pipeline
- Loads multi-asset OHLCV data from Yahoo Finance
- Reads tickers from `src/symbols/portfolio` file (15 crypto assets)
- Calculates returns and technical indicators (moving averages, volatility, momentum)
- Splits data into train/test sets (80/20 by default)
- Loads benchmark indexes for comparison (SPY, BTC-USD)

### Backtesting Features
- Walk-forward validation framework
- Transaction costs simulation (0.1% default)
- Bid-ask spread modeling (0.1% default)
- Portfolio rebalancing costs

### Regime Detection
- Identifies market regimes (Bull/Bear/Sideways)
- Methods: volatility-based, return-based, K-Means, GMM, rule-based
- Computes regime-conditional metrics

### Risk Metrics Analysis
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Annual Return
- Volatility

### Visualizations
1. **Equity Curves** - Comparison of all strategies
2. **Drawdown Analysis** - Maximum drawdown visualization
3. **Rolling Sharpe Ratio** - Time-varying risk-adjusted returns
4. **Rolling Volatility** - Time-varying risk
5. **Portfolio Weights** - Comparison across methods
6. **Metrics Comparison** - Bar charts of risk-adjusted metrics

### Outputs
- CSV files with returns for each strategy
- CSV file with portfolio weights for each method
- CSV file with comparative metrics
- PNG visualization with 6 analysis plots
- Checkpoint file for resumability
- Detailed logs

## Usage

### Basic Usage
```bash
python run/run_pipeline_ensemble_po.py --config configs/portfolio_ensemble.yaml --output outputs/ensemble_po
```

### With Custom Config
```bash
python run/run_pipeline_ensemble_po.py --config configs/my_custom_config.yaml --output outputs/my_run
```

### Resume from Checkpoint
```bash
python run/run_pipeline_ensemble_po.py --config configs/portfolio_ensemble.yaml --output outputs/ensemble_po --resume outputs/ensemble_po/checkpoint.pkl
```

## Configuration

The pipeline is controlled by a YAML configuration file. See `configs/portfolio_ensemble.yaml` for a complete example.

### Key Configuration Options

```yaml
data:
  symbols_file: "src/symbols/portfolio"  # Path to ticker list
  period: "2y"                           # Historical data period
  test_ratio: 0.2                        # Train/test split ratio
  benchmark_tickers: ["SPY", "BTC-USD"]  # Benchmarks for comparison

backtest:
  train_window: 252                      # ~1 year
  test_window: 63                        # ~3 months
  rebalance_freq: 21                     # ~1 month
  transaction_cost: 0.001                # 0.1%
  bid_ask_spread: 0.001                  # 0.1%
  regime_detection:
    method: "volatility"                 # volatility, kmeans, gmm, rule_based
    n_regimes: 3                         # Bull, Bear, Sideways

optimization:
  methods:
    - risk_parity
    - omega
    - cvar
    - hrp
    - efficient_frontier
  params:
    omega:
      target_return: 0.0
    cvar:
      confidence: 0.95
    efficient_frontier:
      risk_aversion: 1.0

display:
  plots:
    - equity_curve
    - drawdown
    - rolling_sharpe
    - rolling_volatility
    - weights_comparison
    - metrics_comparison
  export_format:
    - csv
    - png
  run_id: "ensemble_po_v1"
```

## Portfolio Assets

The pipeline uses the following 15 crypto assets from `src/symbols/portfolio`:

1. ADA-USD (Cardano)
2. BTC-USD (Bitcoin)
3. DOGE-USD (Dogecoin)
4. DOT-USD (Polkadot)
5. ETH-USD (Ethereum)
6. GIGA30063-USD
7. LINK-USD (Chainlink)
8. LTC-USD (Litecoin)
9. PNUT-USD
10. SHIB-USD (Shiba Inu)
11. SOL-USD (Solana)
12. SPX28081-USD
13. SUI20947-USD
14. SUSHI-USD (SushiSwap)
15. UNI7083-USD (Uniswap)

## Example Results

From the test run on 2026-01-30:

### Comparative Metrics
| Strategy | Sharpe | Sortino | Max DD | Calmar | Annual Return | Volatility |
|----------|--------|---------|--------|--------|---------------|------------|
| **risk_parity** | **-1.417** | -2.563 | 0.350 | -2.209 | -0.772 | 0.545 |
| omega | -1.431 | -2.600 | 0.533 | -2.919 | -1.554 | 1.086 |
| cvar | -1.727 | -2.425 | **0.202** | -2.782 | -0.563 | **0.326** |
| hrp | -1.459 | -2.687 | 0.344 | -2.269 | -0.781 | 0.535 |
| efficient_frontier | -1.683 | -2.534 | 0.265 | -2.815 | -0.745 | 0.442 |

**Best Strategy by Sharpe Ratio:** Risk Parity (-1.417)
**Best Strategy by Max Drawdown:** CVaR (0.202)

### Portfolio Weights (Risk Parity)
The Risk Parity method produced the most balanced allocation:
- BTC-USD: 15.1%
- ETH-USD: 8.4%
- LTC-USD: 7.9%
- DOT-USD: 7.6%
- SOL-USD: 7.3%
- SHIB-USD: 6.9%
- DOGE-USD: 6.7%
- LINK-USD: 6.5%
- UNI7083-USD: 6.1%
- SUI20947-USD: 6.0%
- ADA-USD: 5.6%
- SUSHI-USD: 5.1%
- GIGA30063-USD: 3.7%
- SPX28081-USD: 3.6%
- PNUT-USD: 3.4%

## Implementation Details

### Files Created
1. **`src/models/portfolio_optimization.py`** - Portfolio optimization implementations
   - `RiskParityOptimizer`
   - `OmegaRatioOptimizer`
   - `CVaROptimizer`
   - `HRPOptimizer`
   - `EfficientFrontierOptimizer`
   - `PortfolioWeights` - Standard container for weights

2. **`src/models/forecasting.py`** - Forecasting models (for future ensemble use)
   - `LSTMForecaster` - LSTM time series forecasting
   - `SimpleEnsembleForecaster` - Ensemble wrapper
   - `create_sequences` - Data preparation utility

3. **`src/data/portfolio_pipeline.py`** - Portfolio-specific data pipeline
   - `PortfolioPipeline` - Multi-asset data loading and preprocessing
   - `PortfolioDataBundle` - Extended data container
   - `load_index_data` - Benchmark loading utility

4. **`run/run_pipeline_ensemble_po.py`** - Main pipeline script

5. **`configs/portfolio_ensemble.yaml`** - Configuration file

### Dependencies
- NumPy, Pandas - Data manipulation
- SciPy - Optimization algorithms
- PyTorch - Deep learning (for forecasting models)
- scikit-learn - Clustering (optional, for regime detection)
- Matplotlib - Visualization
- yfinance - Data source (via existing OHLCV module)

### Validation
✅ All models produce valid weights (sum to 1.0, non-negative)
✅ Backtesting applies transaction costs correctly
✅ Metrics calculations are numerically stable
✅ Regime detection works with multiple methods
✅ Visualizations render correctly
✅ CSV exports are properly formatted
✅ Checkpoint/resume functionality works

## Advanced Usage

### Custom Optimization Parameters

You can customize parameters for each optimization method in the config:

```yaml
optimization:
  params:
    omega:
      target_return: 0.02  # 2% threshold
    cvar:
      confidence: 0.99     # 99% confidence level
    efficient_frontier:
      target_return: 0.001 # Target 0.1% daily return
      risk_aversion: 2.0   # Higher risk aversion
```

### Adding New Assets

1. Edit `src/symbols/portfolio` to add/remove tickers
2. Or specify directly in config:
```yaml
data:
  tickers: ["BTC-USD", "ETH-USD", "SOL-USD"]
```

### Changing Regime Detection

```yaml
backtest:
  regime_detection:
    method: "kmeans"     # Use K-Means clustering
    n_regimes: 4         # Detect 4 regimes
    window: 90           # 90-day window
```

## Troubleshooting

### Issue: Data download fails
**Solution:** Check internet connection and ensure tickers are valid Yahoo Finance symbols.

### Issue: Optimization fails
**Solution:** Check that returns data has sufficient samples (>30 days) and no NaN values.

### Issue: Memory error
**Solution:** Reduce the period or number of assets in the configuration.

### Issue: Negative Sharpe ratios
**Note:** This is expected in bear markets. The pipeline still correctly identifies the best relative strategy.

## Future Enhancements

- [ ] Add more portfolio optimization methods (Black-Litterman, Kelly Criterion)
- [ ] Integrate LSTM forecasting predictions into optimization
- [ ] Add Monte Carlo simulation for stress testing
- [ ] Implement rolling rebalancing in backtesting
- [ ] Add support for long/short portfolios
- [ ] Include options/derivatives modeling
- [ ] Add multi-period optimization
- [ ] Support for custom constraints (sector limits, etc.)

## References

- Markowitz, H. (1952). Portfolio Selection. The Journal of Finance.
- Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample.
- Ardia, D., et al. (2017). The Impact of Covariance Misspecification in Risk-Based Portfolios.

## License

MIT License - See project LICENSE file for details.
