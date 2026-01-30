"""
Chronos-T5 One-Shot Prediction for Crypto Data
Loads all CSV files from data/raw/ and generates predictions for Close and daily_returns
Creates visualization plots for each file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from chronos import BaseChronosPipeline
    CHRONOS_AVAILABLE = True
    print("Chronos library imported successfully")
except ImportError as e:
    print(f"Warning: chronos-forecasting not installed: {e}")
    print("Install with: pip install chronos-forecasting")
    CHRONOS_AVAILABLE = False
except Exception as e:
    print(f"Error importing chronos: {e}")
    CHRONOS_AVAILABLE = False


def load_data_file(file_path: Path) -> pd.DataFrame:
    """Load a single CSV file and prepare it."""
    df = pd.read_csv(file_path)
    
    # Ensure date column exists
    date_cols = ['date', 'Date', 'DATE', 'timestamp', 'Timestamp']
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df['date'] = pd.to_datetime(df[date_col])
        df = df.sort_values('date')
    
    return df


def prepare_timeseries_data(df: pd.DataFrame, target_col: str, item_id: str = "series_1"):
    """Prepare data in chronos format."""
    # Create a clean dataframe with required columns
    ts_df = pd.DataFrame({
        'item_id': item_id,
        'timestamp': df['date'] if 'date' in df.columns else range(len(df)),
        'target': df[target_col]
    })
    
    # Remove any NaN values
    ts_df = ts_df.dropna()
    
    return ts_df


def predict_with_chronos(context_df: pd.DataFrame, prediction_length: int = 30, pipeline=None):
    """Generate predictions using Chronos-T5."""
    if not CHRONOS_AVAILABLE:
        print("    Chronos not available, using simple forecast")
        return create_simple_forecast(context_df, prediction_length)
    
    try:
        # Use provided pipeline or create new one
        if pipeline is None:
            print("    Loading Chronos-T5 pipeline (this may take a while)...")
            pipeline = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-t5-large",
                device_map="cuda"
            )
            print("    Pipeline loaded successfully")
        
        # Generate predictions
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="item_id",
            timestamp_column="timestamp",
            target="target"
        )
        
        return pred_df, pipeline
    
    except Exception as e:
        print(f"    Chronos prediction failed: {e}")
        print(f"    Using simple forecast instead")
        return create_simple_forecast(context_df, prediction_length), pipeline


def create_simple_forecast(context_df: pd.DataFrame, prediction_length: int):
    """Create a simple forecast when Chronos is not available."""
    # Use last value + random walk or moving average
    target_values = context_df['target'].values
    
    # Simple exponential smoothing
    last_value = target_values[-1]
    trend = np.mean(np.diff(target_values[-30:])) if len(target_values) > 30 else 0
    
    # Generate forecast
    forecast = []
    for i in range(prediction_length):
        next_val = last_value + trend * (i + 1) + np.random.normal(0, np.std(target_values) * 0.1)
        forecast.append(next_val)
    
    # Create prediction dataframe in similar format
    last_timestamp = context_df['timestamp'].iloc[-1]
    
    pred_df = pd.DataFrame({
        'item_id': context_df['item_id'].iloc[0],
        'timestamp': pd.date_range(start=last_timestamp, periods=prediction_length+1, freq='D')[1:],
        'mean': forecast,
        '0.1': [f * 0.95 for f in forecast],
        '0.5': forecast,
        '0.9': [f * 1.05 for f in forecast]
    })
    
    return pred_df


def plot_predictions(
    historical_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    target_col: str,
    symbol: str,
    output_path: Path
):
    """Create and save visualization of historical data and predictions."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Extract historical data
    hist_dates = historical_df['date'] if 'date' in historical_df.columns else range(len(historical_df))
    hist_values = historical_df[target_col].values
    
    # Top plot: Full view
    ax1.plot(hist_dates, hist_values, label='Historical', color='blue', linewidth=1.5, alpha=0.7)
    
    # Plot predictions
    if 'timestamp' in pred_df.columns:
        pred_dates = pred_df['timestamp']
        
        # Plot median prediction
        if '0.5' in pred_df.columns:
            ax1.plot(pred_dates, pred_df['0.5'], label='Forecast (Median)', 
                    color='red', linewidth=2, linestyle='--')
        elif 'mean' in pred_df.columns:
            ax1.plot(pred_dates, pred_df['mean'], label='Forecast (Mean)', 
                    color='red', linewidth=2, linestyle='--')
        
        # Plot confidence intervals
        if '0.1' in pred_df.columns and '0.9' in pred_df.columns:
            ax1.fill_between(pred_dates, pred_df['0.1'], pred_df['0.9'], 
                           color='red', alpha=0.2, label='80% Confidence Interval')
    
    ax1.set_title(f'{symbol} - {target_col} - Full History with Forecast', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(target_col)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Zoomed to last 90 days + predictions
    zoom_days = min(90, len(hist_dates))
    zoom_hist_dates = hist_dates[-zoom_days:] if isinstance(hist_dates, pd.Series) else list(range(len(hist_dates)))[-zoom_days:]
    zoom_hist_values = hist_values[-zoom_days:]
    
    ax2.plot(zoom_hist_dates, zoom_hist_values, label='Historical', 
            color='blue', linewidth=2, alpha=0.7, marker='o', markersize=3)
    
    # Plot predictions on zoom
    if 'timestamp' in pred_df.columns:
        if '0.5' in pred_df.columns:
            ax2.plot(pred_dates, pred_df['0.5'], label='Forecast (Median)', 
                    color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
        elif 'mean' in pred_df.columns:
            ax2.plot(pred_dates, pred_df['mean'], label='Forecast (Mean)', 
                    color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
        
        if '0.1' in pred_df.columns and '0.9' in pred_df.columns:
            ax2.fill_between(pred_dates, pred_df['0.1'], pred_df['0.9'], 
                           color='red', alpha=0.2, label='80% Confidence Interval')
    
    ax2.set_title(f'Last {zoom_days} Days + Forecast (Zoomed)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel(target_col)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {output_path}")


def process_all_files():
    """Main function to process all CSV files."""
    global CHRONOS_AVAILABLE
    
    # Setup paths
    data_dir = Path("data/raw")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Get all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    print("=" * 80)
    
    today = datetime.now().strftime("%Y%m%d")
    prediction_length = 30  # Forecast 30 days ahead
    
    # Load Chronos pipeline once if available
    pipeline = None
    use_chronos = CHRONOS_AVAILABLE
    
    if use_chronos:
        try:
            print("\nLoading Chronos-T5 pipeline (this may take a while)...")
            pipeline = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-t5-large",
                device_map="cuda"
            )
            print("Pipeline loaded successfully!\n")
        except Exception as e:
            print(f"Failed to load Chronos pipeline: {e}")
            print("Will use simple forecasting instead\n")
            use_chronos = False
    
    # Process each file
    for i, file_path in enumerate(csv_files, 1):
        symbol = file_path.stem  # Get filename without extension
        print(f"\n[{i}/{len(csv_files)}] Processing: {symbol}")
        
        try:
            # Load data
            df = load_data_file(file_path)
            print(f"  Loaded {len(df)} rows")
            
            # Check for required columns
            target_columns = []
            if 'close' in df.columns:
                target_columns.append('close')
            if 'daily_returns' in df.columns:
                target_columns.append('daily_returns')
            
            if not target_columns:
                print(f"  Warning: No 'close' or 'daily_returns' columns found. Available: {list(df.columns)}")
                continue
            
            print(f"  Target columns: {target_columns}")
            
            # Process each target column
            for target_col in target_columns:
                print(f"\n  Forecasting: {target_col}")
                
                # Prepare data for Chronos
                ts_df = prepare_timeseries_data(df, target_col, item_id=symbol)
                
                if len(ts_df) < 10:
                    print(f"    Warning: Only {len(ts_df)} valid data points. Skipping.")
                    continue
                
                print(f"    Historical data points: {len(ts_df)}")
                
                # Generate predictions
                if use_chronos and pipeline is not None:
                    result = predict_with_chronos(ts_df, prediction_length, pipeline)
                    if isinstance(result, tuple):
                        pred_df, _ = result
                    else:
                        pred_df = result
                else:
                    pred_df = create_simple_forecast(ts_df, prediction_length)
                
                if pred_df is not None and len(pred_df) > 0:
                    print(f"    Generated {len(pred_df)} forecast points")
                    
                    # Create plot
                    plot_filename = f"{symbol}_{target_col}_{today}.png"
                    plot_path = plots_dir / plot_filename
                    
                    plot_predictions(df, pred_df, target_col, symbol, plot_path)
                else:
                    print(f"    Warning: Prediction failed for {target_col}")
        
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print(f"Processing complete! Plots saved to {plots_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("Chronos-T5 Crypto Prediction Pipeline")
    print("=" * 80)
    
    if not CHRONOS_AVAILABLE:
        print("\nNote: Running in fallback mode (simple forecasting)")
        print("For Chronos-T5 predictions, install: pip install chronos-forecasting\n")
    else:
        print("\nChronos-T5 model loaded successfully\n")
    
    process_all_files()
