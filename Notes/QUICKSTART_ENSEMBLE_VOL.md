# Quick Start Guide: Ensemble Volatility Forecasting

## 1-Minute Setup

### Prerequisites
```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Optional: For full GARCH functionality
pip install arch
```

### Run the Pipeline
```bash
# From project root directory
python run/run_pipeline_ensemble_vol.py --config configs/volatility_ensemble.yaml --output outputs/vol_forecast
```

## What Happens

1. **Data Loading** (2-5 seconds)
   - Loads 15 crypto assets from `src/symbols/portfolio`
   - Downloads 2 years of OHLCV data
   - Splits into 80% train / 20% test

2. **Realized Volatility Calculation** (<1 second)
   - Computes 3 volatility estimators:
     * Standard (close-to-close)
     * Parkinson (high-low)
     * Garman-Klass (OHLC)

3. **Volatility Characteristics** (<1 second)
   - Analyzes persistence (autocorrelation)
   - Measures asymmetry (leverage effect)
   - Calculates half-life of mean reversion

4. **Model Training** (5-15 seconds)
   - **GARCH(1,1)**: ~2 seconds
   - **HAR**: <1 second
   - **LSTM**: 5-10 seconds (GPU accelerated if available)

5. **Forecasting** (<1 second)
   - Generates volatility forecasts for test period
   - Creates ensemble forecast (average of all models)

6. **Evaluation** (<1 second)
   - Computes 15+ forecast metrics
   - Runs Diebold-Mariano test
   - Calculates VaR coverage

7. **Visualizations** (1-2 seconds)
   - Generates 9-panel analysis chart
   - Shows forecasts, errors, distributions, surface

8. **Export** (<1 second)
   - Saves CSV files with metrics, forecasts, characteristics
   - Creates checkpoint for resumability

**Total Time: ~15-30 seconds**

## Check Your Results

```bash
# View metrics
cat outputs/vol_forecast/vol_forecast_v1_metrics.csv

# View forecasts
cat outputs/vol_forecast/vol_forecast_v1_forecasts.csv

# View volatility characteristics
cat outputs/vol_forecast/vol_forecast_v1_characteristics.csv

# View visualization
# Open: outputs/vol_forecast/vol_forecast_v1_analysis.png

# View logs
cat outputs/vol_forecast/logs/vol_forecast_v1.log
```

## Understanding Your Results

### Metrics Output
```
Model     RMSE    MAE     R²      QLIKE   Direction%
HAR       0.151   0.127   -2.57   0.022   51.9%
Ensemble  0.194   0.160   -4.89   0.032   46.8%
LSTM      0.215   0.177   -5.45   0.049   52.7%
GARCH     0.301   0.291   -13.22  0.065   6.5%
```

**What to look for:**
- ✅ **Lowest RMSE/MAE** = Best forecast accuracy
- ✅ **Lowest QLIKE** = Best asymmetric loss (volatility-specific)
- ✅ **Direction >50%** = Better than random guess
- ⚠️ **Negative R²** = Normal for volatility (inherently noisy)

### Volatility Characteristics
```
Ticker    Persistence   Half-life   Asymmetry
ADA-USD   0.963        18.3 days   0.981
BTC-USD   0.977        29.7 days   0.998
```

**Interpretation:**
- **Persistence >0.95** = Strong volatility clustering
- **Half-life** = Days for shock to decay by 50%
- **Asymmetry near 1.0** = Symmetric response (no leverage effect)
- **Asymmetry <1.0** = Negative returns → higher volatility (leverage)

## Customize Your Run

### Change Assets
Edit `src/symbols/portfolio` or specify in config:
```yaml
data:
  tickers: ["BTC-USD", "ETH-USD", "SOL-USD"]
```

### Change Time Period
```yaml
data:
  period: "5y"  # Use 5 years instead of 2
```

### Add More Models
```yaml
models:
  types:
    - garch
    - egarch  # Add Exponential GARCH
    - har
    - lstm
```

### Tune LSTM
```yaml
models:
  lstm:
    seq_length: 44    # Use ~2 months of data
    hidden_size: 64   # Bigger network
    epochs: 100       # Train longer
```

### Change Forecast Horizon
```yaml
forecasting:
  horizon: 5  # Forecast 5 days ahead instead of 1
```

## Expected Output

### Console Output
```
Starting Ensemble Volatility Forecasting Pipeline
Loaded 15 assets
Training samples: 311
Testing samples: 78

Training GARCH(1,1) model...
  Omega: 0.003332
  Alpha: 0.064953
  Beta: 0.013070
  Persistence: 0.078023

Training HAR model...
  Intercept: 0.058184
  Daily coef: 1.042397

Training LSTM model...
  Training completed. Final loss: 0.017453

Best Model: HAR
Best RMSE: 0.1509
PIPELINE COMPLETED SUCCESSFULLY
```

### Files Created
```
outputs/vol_forecast/
├── vol_forecast_v1_analysis.png          # 9-panel visualization
├── vol_forecast_v1_metrics.csv           # Model comparison metrics
├── vol_forecast_v1_forecasts.csv         # All forecasts + actual
├── vol_forecast_v1_characteristics.csv   # Volatility properties
├── vol_forecast_v1_checkpoint.pkl        # Resume checkpoint
└── logs/
    └── vol_forecast_v1.log              # Detailed logs
```

## Interpreting the Visualization

The 9-panel chart shows:

1. **Top-Left**: Time series of all model forecasts vs actual
2. **Top-Middle**: Forecast errors (residuals) over time
3. **Top-Right**: Scatter plot (actual vs predicted)
4. **Middle-Left**: Comparison of different volatility estimators
5. **Middle-Center**: Volatility cone (historical distribution)
6. **Middle-Right**: Bar chart comparing model metrics
7. **Bottom-Left**: Multi-asset volatility surface (heatmap)
8. **Bottom-Center**: Returns vs volatility (dual-axis)
9. **Bottom-Right**: Distribution comparison (histograms)

## Common Patterns

### Good Forecast
- RMSE < 0.20
- Direction accuracy > 55%
- QLIKE < 0.05
- Predictions track actual closely

### Poor Forecast
- RMSE > 0.40
- Direction accuracy < 45%
- QLIKE > 0.10
- Flat predictions (doesn't capture dynamics)

### Model Selection
- **HAR often wins** for daily volatility
- **LSTM good** for complex patterns
- **GARCH good** for very short-term (intraday)
- **Ensemble robust** across different regimes

## Troubleshooting

**Error: "arch module not found"**
→ Install: `pip install arch` OR pipeline will use fallback implementation

**Warning: "GARCH training failed"**
→ Normal for some assets. Pipeline continues with other models.

**Slow performance?**
→ Reduce `epochs` in LSTM config or use fewer assets

**Negative R²?**
→ This is normal for volatility forecasting. Focus on RMSE and QLIKE instead.

## Next Steps

1. **Compare Different Assets**
   ```bash
   # Edit src/symbols/portfolio to include different tickers
   python run/run_pipeline_ensemble_vol.py --config configs/volatility_ensemble.yaml --output outputs/stocks_vol
   ```

2. **Test Different Time Periods**
   ```bash
   # Create custom config with period: "1y"
   python run/run_pipeline_ensemble_vol.py --config configs/short_period.yaml --output outputs/1year_vol
   ```

3. **Optimize LSTM Hyperparameters**
   - Try different `seq_length` values (10, 22, 44)
   - Adjust `hidden_size` (16, 32, 64, 128)
   - Experiment with `num_layers` (1, 2, 3)

4. **Use Forecasts for Trading**
   - High volatility → reduce position size
   - Low volatility → increase position size
   - Rising volatility → potential market stress

## Performance Tips

✅ **Use GPU** - LSTM training 3-5x faster
✅ **Limit assets** - Start with 5-10 for faster iteration
✅ **Reduce epochs** - 30-50 often sufficient for testing
✅ **Use fewer estimators** - Standard + Garman-Klass usually enough
✅ **Focus on primary ticker** - Detailed analysis on one asset

## When to Use Each Model

### GARCH
- Very short-term forecasts (1-3 days)
- High-frequency data
- When you need parameter interpretation
- Regulated environments (standard model)

### HAR
- Daily to weekly forecasts
- Simple and robust
- Good baseline
- When data is limited

### LSTM
- Complex patterns
- Multiple input features
- Non-linear relationships
- When you have GPU and data

### Ensemble
- Want robustness
- Uncertain which model best
- Production systems
- Risk management applications

## Success Metrics

### Research Quality
- RMSE within 0.10-0.30 (depends on asset)
- Direction accuracy >50%
- VaR coverage 90-98%
- Forecasts track major volatility spikes

### Production Quality
- RMSE consistently <0.25
- Direction accuracy >55%
- VaR coverage 93-97%
- Stable performance across regimes
- Fast inference (<100ms)

---

**Ready to forecast volatility? Run the command above and check your results in ~20 seconds!**
