# Quick Start Guide: Ensemble Portfolio Optimization

## 1-Minute Setup

### Prerequisites
```bash
# Ensure dependencies are installed
pip install -r requirements.txt
```

### Run the Pipeline
```bash
# From project root directory
python run/run_pipeline_ensemble_po.py --config configs/portfolio_ensemble.yaml --output outputs/my_first_run
```

## What Happens

1. **Data Loading** (5-10 seconds)
   - Loads 15 crypto assets from `src/symbols/portfolio`
   - Downloads 2 years of historical data from Yahoo Finance
   - Splits into 80% train / 20% test

2. **Portfolio Optimization** (<1 second)
   - Computes optimal weights using 5 different methods:
     * Risk Parity
     * Omega Ratio
     * CVaR
     * Hierarchical Risk Parity (HRP)
     * Efficient Frontier

3. **Backtesting** (<1 second)
   - Simulates portfolio performance on test period
   - Applies transaction costs (0.1% default)
   - Records returns for each strategy

4. **Regime Detection** (<1 second)
   - Identifies Bull/Bear/Sideways market regimes
   - Computes regime-conditional metrics

5. **Metrics & Analysis** (<1 second)
   - Calculates Sharpe, Sortino, Max Drawdown, Calmar
   - Compares strategies side-by-side

6. **Visualizations** (<1 second)
   - Generates 6-panel analysis chart
   - Shows equity curves, drawdowns, rolling metrics

7. **Export** (<1 second)
   - Saves CSV files with returns, weights, metrics
   - Creates checkpoint for resumability

**Total Time: ~20 seconds**

## Check Your Results

```bash
# View comparison metrics
cat outputs/my_first_run/ensemble_po_v1_comparison.csv

# View portfolio weights
cat outputs/my_first_run/ensemble_po_v1_weights.csv

# View visualization
# Open: outputs/my_first_run/ensemble_po_v1_analysis.png

# View detailed logs
cat outputs/my_first_run/logs/ensemble_po_v1.log
```

## Customize Your Run

### Change Time Period
Edit `configs/portfolio_ensemble.yaml`:
```yaml
data:
  period: "5y"  # Use 5 years instead of 2
```

### Change Assets
Edit `src/symbols/portfolio` or specify in config:
```yaml
data:
  tickers: ["BTC-USD", "ETH-USD", "SOL-USD"]
```

### Add More Optimization Methods
Edit `configs/portfolio_ensemble.yaml`:
```yaml
optimization:
  methods:
    - risk_parity
    - omega
    - cvar
    - hrp
    - efficient_frontier
```

### Adjust Transaction Costs
```yaml
backtest:
  transaction_cost: 0.002  # 0.2%
  bid_ask_spread: 0.002    # 0.2%
```

## Expected Output

### Console Output
```
Starting Ensemble Portfolio Optimization Pipeline
Loaded 15 assets
Training samples: 337
Testing samples: 85
Optimizing with method: risk_parity
  risk_parity weights: {'BTC-USD': 0.151, 'ETH-USD': 0.084, ...}
Backtesting strategy: risk_parity
  Total return: -0.260
  Transaction cost: 0.0003
...
Best Strategy: risk_parity
Best Sharpe Ratio: -1.417
PIPELINE COMPLETED SUCCESSFULLY
```

### Files Created
```
outputs/my_first_run/
├── ensemble_po_v1_analysis.png          # 6-panel visualization
├── ensemble_po_v1_comparison.csv        # Metrics comparison table
├── ensemble_po_v1_weights.csv           # Portfolio weights
├── ensemble_po_v1_risk_parity_returns.csv
├── ensemble_po_v1_omega_returns.csv
├── ensemble_po_v1_cvar_returns.csv
├── ensemble_po_v1_hrp_returns.csv
├── ensemble_po_v1_efficient_frontier_returns.csv
├── ensemble_po_v1_checkpoint.pkl        # Resume checkpoint
└── logs/
    └── ensemble_po_v1.log               # Detailed logs
```

## Understanding Results

### Best Strategy Selection
The pipeline identifies the best strategy by **Sharpe Ratio** (risk-adjusted return).

### Negative Returns?
This is normal in bear markets! The pipeline still identifies which strategy loses *least*.

### Metrics Explained

- **Sharpe Ratio**: Return per unit of risk (higher is better)
  - \> 1.0 = Good
  - \> 1.5 = Excellent
  - \> 2.0 = Outstanding
  
- **Sortino Ratio**: Like Sharpe but only penalizes downside risk
  
- **Max Drawdown**: Largest peak-to-trough decline (lower is better)
  - < 0.10 = 10% max loss (good)
  - < 0.20 = 20% max loss (acceptable)
  - \> 0.30 = 30%+ max loss (risky)
  
- **Calmar Ratio**: Annual return / Max drawdown (higher is better)

- **Annual Return**: Expected yearly return (252 trading days)

- **Volatility**: Standard deviation of returns (lower is better)

## Next Steps

1. **Compare Multiple Periods**
   ```bash
   python run/run_pipeline_ensemble_po.py --config configs/portfolio_ensemble.yaml --output outputs/run_2020_2022
   python run/run_pipeline_ensemble_po.py --config configs/portfolio_ensemble.yaml --output outputs/run_2022_2024
   ```

2. **Test Different Regimes**
   Change regime detection method in config:
   ```yaml
   backtest:
     regime_detection:
       method: "kmeans"  # Try: volatility, kmeans, gmm, rule_based
   ```

3. **Add Your Own Assets**
   Create `src/symbols/my_portfolio` with your tickers

4. **Optimize Parameters**
   Adjust CVaR confidence, Omega target, etc. in config

## Troubleshooting

**Error: "No tickers found"**
→ Check that `src/symbols/portfolio` exists and has valid tickers

**Error: "Data download failed"**
→ Check internet connection, ensure tickers are valid Yahoo Finance symbols

**Warning: "Optimization failed for method X"**
→ Usually means insufficient data or numerical instability, pipeline continues with other methods

**Slow performance?**
→ Reduce `period` in config or use fewer assets

## Support

For issues or questions:
1. Check `run/README_ENSEMBLE_PO.md` for detailed documentation
2. Review logs in `outputs/<run_name>/logs/`
3. Examine checkpoint file for debugging: `outputs/<run_name>/*.pkl`

## Tips for Best Results

✅ Use at least 2 years of data for reliable statistics
✅ Include 8-20 assets for good diversification
✅ Use crypto-to-crypto comparisons (all crypto or all stocks, not mixed)
✅ Adjust transaction costs to match your trading platform
✅ Run multiple backtests on different time periods
✅ Compare against benchmark (SPY for stocks, BTC-USD for crypto)

## Performance Notes

- **Data Download**: 5-10 seconds (depends on internet speed and # of assets)
- **Optimization**: <1 second per method
- **Backtesting**: <1 second (scales linearly with test period length)
- **Visualizations**: <1 second
- **Total Runtime**: Typically 15-30 seconds

For 100+ assets or 10+ years of data, expect 1-2 minutes.

---

**Ready to optimize your portfolio? Run the command above and check your results in ~20 seconds!**
