# Crypto Chronos-T5 Prediction Script

## Overview

`crypto_chronos_t5.py` is an automated forecasting script that processes all CSV files in `data/raw/` and generates one-shot predictions using the Chronos-T5 model (or a simple fallback method) for both `close` prices and `daily_returns`.

## Features

- **Batch Processing**: Automatically processes all CSV files in `data/raw/`
- **Dual Predictions**: Forecasts both `close` prices and `daily_returns` for each dataset
- **Smart Fallback**: Uses Chronos-T5 when available, falls back to simple exponential smoothing
- **Comprehensive Visualization**: Creates 2-panel plots showing:
  - Full historical data + forecast
  - Zoomed view (last 90 days + forecast)
- **Confidence Intervals**: Shows 80% confidence bands (10th to 90th percentile)
- **Robust Error Handling**: Continues processing even if individual files fail

## Usage

```bash
# Basic usage
python scripts/crypto_chronos_t5.py

# Runtime: ~20-30 seconds for 13 files (with Chronos-T5 loaded)
```

## Requirements

### Optional (for Chronos-T5):
```bash
pip install chronos-forecasting
```

### Standard Libraries:
- pandas
- numpy
- matplotlib
- pathlib

## Output

### Files Created
- **Location**: `plots/`
- **Format**: `{symbol}_{target}_{date}.png`
- **Count**: 2 plots per target column per CSV file

### Example Run Results

**Processed**: 13 CSV files
**Generated**: 23 plots

| File | Close Plots | Daily Returns Plots |
|------|-------------|---------------------|
| AAVE-GBP_20260130.csv | ✅ | ✅ |
| BCH-AUD_20260130.csv | ✅ | ✅ |
| ETH-JPY_20260130.csv | ✅ | ✅ |
| GOAT-USD_20260130.csv | ✅ | ✅ |
| LSETH-USD_20260130.csv | ✅ | ✅ |
| EUR_CAD_20260130.csv | ✅ | ✅ |
| GNO_USD_20260130.csv | ✅ | ✅ |
| KEEP_USD_20260130.csv | ✅ | ✅ |
| TIA_USD_20260130.csv | ✅ | ✅ |
| XMN_USD_20260130.csv | ✅ | ✅ |
| BAMLC0A4CBBB_20260130.csv | N/A | ✅ |
| UMCSENT_20260130.csv | N/A | ✅ |
| CPIAUCSL_20260130.csv | N/A | ✅ |

## Plot Structure

Each plot contains:

### Top Panel: Full History View
- Complete historical data (blue line)
- 30-day forecast (red dashed line)
- 80% confidence interval (shaded red area)
- Clear legend and grid

### Bottom Panel: Zoomed View
- Last 90 days of historical data (blue line with markers)
- 30-day forecast (red dashed line with markers)
- 80% confidence interval (shaded red area)
- Enhanced visibility for recent trends

## Data Requirements

CSV files must contain:
- **Required**: A date column (`date`, `Date`, `timestamp`, etc.)
- **Target columns**: At least one of:
  - `close` - Price data
  - `daily_returns` - Return data

Files are automatically sorted by date and cleaned of NaN values.

## Forecast Method

### With Chronos-T5 (Preferred)
- **Model**: amazon/chronos-t5-large
- **Method**: Probabilistic forecasting with quantiles
- **Horizon**: 30 days
- **Quantiles**: 10%, 50%, 90%
- **GPU Accelerated**: Uses CUDA if available

### Fallback (Simple Forecast)
- **Method**: Exponential smoothing with trend
- **Features**:
  - Captures recent trend from last 30 days
  - Adds realistic noise based on historical volatility
  - Generates confidence intervals (±5%)

## Error Handling

The script handles several error scenarios:

1. **Chronos Import Failure**: Falls back to simple forecasting
2. **Chronos Model Loading Failure**: Falls back to simple forecasting
3. **Frequency Inference Failure**: Falls back to simple forecasting
4. **Missing Columns**: Skips file with warning
5. **Insufficient Data**: Skips target with warning (<10 data points)
6. **General Errors**: Logs error and continues to next file

## Performance

### With Chronos-T5
- **Model Loading**: ~5-10 seconds (one-time)
- **Per File**: <1 second per prediction
- **Total Runtime**: ~20-30 seconds for 13 files

### Simple Fallback
- **No Model Loading**
- **Per File**: <0.1 seconds per prediction
- **Total Runtime**: ~5 seconds for 13 files

## Example Output

```
================================================================================
Chronos-T5 Crypto Prediction Pipeline
================================================================================

Found 13 CSV files to process

Loading Chronos-T5 pipeline (this may take a while)...
Pipeline loaded successfully!

[1/13] Processing: AAVE-GBP_20260130
  Loaded 1096 rows
  Target columns: ['close', 'daily_returns']
  
  Forecasting: close
    Historical data points: 1096
    Generated 30 forecast points
  Saved plot: plots\AAVE-GBP_20260130_close_20260130.png
  
  Forecasting: daily_returns
    Historical data points: 1096
    Generated 30 forecast points
  Saved plot: plots\AAVE-GBP_20260130_daily_returns_20260130.png

[2/13] Processing: BCH-AUD_20260130
  ...

================================================================================
Processing complete! Plots saved to plots/
================================================================================
```

## Customization

### Change Forecast Horizon
```python
prediction_length = 60  # Forecast 60 days instead of 30
```

### Change Confidence Intervals
```python
quantile_levels=[0.05, 0.5, 0.95]  # 90% confidence instead of 80%
```

### Change Zoom Window
```python
zoom_days = 120  # Show last 120 days instead of 90
```

## Troubleshooting

### Issue: "chronos-forecasting not installed"
**Solution**: 
```bash
pip install chronos-forecasting
```
Or script will automatically use simple forecasting.

### Issue: "Could not infer frequency"
**Note**: This is normal for some datasets. Script automatically falls back to simple forecasting.

### Issue: CUDA out of memory
**Solution**: 
```python
device_map="cpu"  # Use CPU instead of GPU
```

### Issue: Plots not showing recent data
**Note**: Check if CSV files contain recent dates. Script uses actual dates from data.

## Notes

- The script is designed to be robust and continue processing even if individual predictions fail
- Chronos-T5 works best with regular time series (daily frequency)
- For irregular time series, the simple fallback method is used
- All plots are saved with unique names including the date stamp
- GPU acceleration significantly speeds up Chronos-T5 predictions

## Future Enhancements

- [ ] Add support for custom forecast horizons per file
- [ ] Include multiple confidence levels
- [ ] Add ensemble forecasting (combine Chronos + other methods)
- [ ] Export forecast data to CSV
- [ ] Add interactive plots with Plotly
- [ ] Support for minute/hourly frequency data

## License

MIT License - See project LICENSE file for details.
