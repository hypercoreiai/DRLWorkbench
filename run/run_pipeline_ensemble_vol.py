"""
Ensemble Volatility Forecasting Pipeline
Combines multiple volatility models (GARCH, EGARCH, HAR, LSTM) to forecast asset volatility
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils import ConfigError
from src.data.volatility_pipeline import VolatilityPipeline, create_vol_sequences
from src.models.volatility_forecasting import (
    get_volatility_model,
    RealizedVolatility,
    GARCHForecaster,
    HARForecaster,
    LSTMVolForecaster
)
from src.analysis.volatility_metrics import (
    compute_volatility_forecast_metrics,
    compute_volatility_persistence,
    compute_volatility_asymmetry,
    compute_diebold_mariano_test
)
from src.display.volatility_plots import (
    plot_volatility_forecast,
    plot_volatility_forecast_errors,
    plot_volatility_scatter,
    plot_realized_volatility_comparison,
    plot_volatility_surface,
    plot_volatility_cone,
    plot_volatility_term_structure,
    plot_model_comparison_metrics
)

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return config dict. Raises ConfigError on failure."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML: {e}") from e


def run_volatility_forecasting_pipeline(
    config_path: str,
    output_dir: str,
    resume: Optional[str] = None,
) -> None:
    """
    Main pipeline for ensemble volatility forecasting.
    
    Steps:
    1. Load data and calculate realized volatility
    2. Train multiple volatility models (GARCH, HAR, LSTM)
    3. Generate forecasts
    4. Evaluate forecast accuracy
    5. Analyze volatility characteristics
    6. Create comprehensive visualizations
    7. Export results
    
    Args:
        config_path: Path to configuration YAML file.
        output_dir: Directory for outputs.
        resume: Optional checkpoint path to resume from.
    """
    # Load configuration
    config = load_config(config_path)
    run_id = config.get("display", {}).get("run_id", "vol_forecast")
    
    # Setup output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(str(log_dir), run_id)
    logger.info("=" * 80)
    logger.info("Starting Ensemble Volatility Forecasting Pipeline")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    
    # State for checkpointing
    state: Dict[str, Any] = {}
    if resume:
        try:
            state = load_checkpoint(resume)
            logger.info(f"Resumed from checkpoint: {resume}")
        except FileNotFoundError:
            logger.warning(f"Checkpoint not found, starting fresh: {resume}")
    
    # =====================================================================
    # STEP 1: DATA PIPELINE
    # =====================================================================
    logger.info("=" * 80)
    logger.info("STEP 1: Data Pipeline & Realized Volatility")
    logger.info("=" * 80)
    
    pipeline = VolatilityPipeline(config)
    data_bundle = pipeline.run()
    
    if data_bundle.realized_vol_train is None or data_bundle.realized_vol_train.empty:
        logger.error("Data pipeline failed to produce training data")
        return
    
    tickers = data_bundle.metadata['tickers']
    vol_estimators = data_bundle.metadata['vol_estimators']
    
    logger.info(f"Loaded {len(tickers)} assets: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")
    logger.info(f"Training samples: {len(data_bundle.realized_vol_train)}")
    logger.info(f"Testing samples: {len(data_bundle.realized_vol_test)}")
    logger.info(f"Volatility estimators: {', '.join(vol_estimators)}")
    
    # =====================================================================
    # STEP 2: VOLATILITY CHARACTERISTICS ANALYSIS
    # =====================================================================
    logger.info("=" * 80)
    logger.info("STEP 2: Volatility Characteristics Analysis")
    logger.info("=" * 80)
    
    vol_characteristics = {}
    
    for ticker in tickers[:3]:  # Analyze first 3 tickers in detail
        logger.info(f"\nAnalyzing {ticker}:")
        
        # Get realized volatility for this ticker (using standard estimator)
        vol_col = f"{ticker}_standard"
        if vol_col in data_bundle.realized_vol_train.columns:
            rv_series = pd.concat([
                data_bundle.realized_vol_train[vol_col],
                data_bundle.realized_vol_test[vol_col]
            ])
            
            # Persistence
            persistence = compute_volatility_persistence(rv_series)
            logger.info(f"  Persistence (ACF lag-1): {persistence['acf_lag1']:.3f}")
            logger.info(f"  Half-life: {persistence['half_life']:.1f} days")
            
            # Asymmetry / Leverage effect
            returns = pd.concat([
                data_bundle.returns_train[ticker],
                data_bundle.returns_test[ticker]
            ])
            asymmetry = compute_volatility_asymmetry(returns, rv_series)
            logger.info(f"  Asymmetry ratio: {asymmetry['asymmetry_ratio']:.3f}")
            logger.info(f"  Vol after negative returns: {asymmetry['vol_after_negative_ret']:.3f}")
            logger.info(f"  Vol after positive returns: {asymmetry['vol_after_positive_ret']:.3f}")
            
            vol_characteristics[ticker] = {
                'persistence': persistence,
                'asymmetry': asymmetry
            }
    
    # =====================================================================
    # STEP 3: MODEL TRAINING
    # =====================================================================
    logger.info("=" * 80)
    logger.info("STEP 3: Model Training")
    logger.info("=" * 80)
    
    models_config = config.get('models', {})
    model_types = models_config.get('types', ['garch', 'har', 'lstm'])
    
    trained_models = {}
    
    # Select primary ticker for detailed modeling
    primary_ticker = tickers[0]
    logger.info(f"\nTraining models for primary ticker: {primary_ticker}")
    
    # Get data for primary ticker
    returns_train = data_bundle.returns_train[primary_ticker]
    vol_col = f"{primary_ticker}_standard"
    rv_train = data_bundle.realized_vol_train[vol_col]
    rv_test = data_bundle.realized_vol_test[vol_col]
    
    # Train GARCH
    if 'garch' in model_types:
        logger.info("\nTraining GARCH(1,1) model...")
        garch_model = GARCHForecaster()
        try:
            garch_model.fit(returns_train)
            trained_models['GARCH'] = garch_model
            params = garch_model.get_hyperparams()
            logger.info(f"  Omega: {params['omega']:.6f}")
            logger.info(f"  Alpha: {params['alpha']:.6f}")
            logger.info(f"  Beta: {params['beta']:.6f}")
            logger.info(f"  Persistence: {params['persistence']:.6f}")
        except Exception as e:
            logger.error(f"  GARCH training failed: {e}")
    
    # Train HAR
    if 'har' in model_types:
        logger.info("\nTraining HAR model...")
        har_model = HARForecaster()
        try:
            har_model.fit(rv_train)
            trained_models['HAR'] = har_model
            params = har_model.get_hyperparams()
            logger.info(f"  Intercept: {params['intercept']:.6f}")
            logger.info(f"  Daily coef: {params['coef_daily']:.6f}")
            logger.info(f"  Weekly coef: {params['coef_weekly']:.6f}")
            logger.info(f"  Monthly coef: {params['coef_monthly']:.6f}")
        except Exception as e:
            logger.error(f"  HAR training failed: {e}")
    
    # Train LSTM (if configured)
    if 'lstm' in model_types:
        logger.info("\nTraining LSTM model...")
        lstm_model = LSTMVolForecaster()
        try:
            # Create sequences
            seq_length = models_config.get('lstm', {}).get('seq_length', 22)
            
            # Use subset of features
            feature_cols = [col for col in data_bundle.features_train.columns 
                          if primary_ticker in col][:10]  # Limit features
            
            X_train, y_train = create_vol_sequences(
                data_bundle.features_train[feature_cols],
                rv_train,
                seq_length=seq_length,
                forecast_horizon=1
            )
            
            lstm_config = models_config.get('lstm', {})
            lstm_model.fit(X_train, y_train, config=lstm_config)
            
            trained_models['LSTM'] = lstm_model
            logger.info(f"  Training completed. Final loss: {lstm_model.history['train_loss'][-1]:.6f}")
        except Exception as e:
            logger.error(f"  LSTM training failed: {e}")
    
    if not trained_models:
        logger.error("No models were successfully trained")
        return
    
    logger.info(f"\nSuccessfully trained {len(trained_models)} models")
    
    # =====================================================================
    # STEP 4: FORECASTING
    # =====================================================================
    logger.info("=" * 80)
    logger.info("STEP 4: Volatility Forecasting")
    logger.info("=" * 80)
    
    forecast_horizon = config.get('forecasting', {}).get('horizon', 1)
    forecasts = {}
    
    # GARCH forecasts
    if 'GARCH' in trained_models:
        logger.info("\nGenerating GARCH forecasts...")
        try:
            # Use full returns for conditional forecast
            returns_full = pd.concat([returns_train, rv_test.iloc[:0]])  # Up to test start
            garch_pred = trained_models['GARCH'].predict(returns_full, horizon=len(rv_test))
            
            # Create series with test index
            forecasts['GARCH'] = pd.Series(garch_pred, index=rv_test.index[:len(garch_pred)])
            logger.info(f"  Generated {len(garch_pred)} forecasts")
        except Exception as e:
            logger.error(f"  GARCH forecasting failed: {e}")
    
    # HAR forecasts
    if 'HAR' in trained_models:
        logger.info("\nGenerating HAR forecasts...")
        try:
            rv_full = pd.concat([rv_train, rv_test])
            har_pred = trained_models['HAR'].predict(rv_full.iloc[:len(rv_train)], horizon=len(rv_test))
            
            forecasts['HAR'] = pd.Series(har_pred, index=rv_test.index[:len(har_pred)])
            logger.info(f"  Generated {len(har_pred)} forecasts")
        except Exception as e:
            logger.error(f"  HAR forecasting failed: {e}")
    
    # LSTM forecasts
    if 'LSTM' in trained_models:
        logger.info("\nGenerating LSTM forecasts...")
        try:
            X_test, y_test = create_vol_sequences(
                data_bundle.features_test[feature_cols],
                rv_test,
                seq_length=seq_length,
                forecast_horizon=1
            )
            
            lstm_pred = trained_models['LSTM'].predict(X_test).flatten()
            
            forecasts['LSTM'] = pd.Series(lstm_pred, index=rv_test.index[:len(lstm_pred)])
            logger.info(f"  Generated {len(lstm_pred)} forecasts")
        except Exception as e:
            logger.error(f"  LSTM forecasting failed: {e}")
    
    # Ensemble forecast (simple average)
    if len(forecasts) > 1:
        logger.info("\nCreating ensemble forecast...")
        forecast_df = pd.DataFrame(forecasts)
        ensemble_pred = forecast_df.mean(axis=1)
        forecasts['Ensemble'] = ensemble_pred
        logger.info(f"  Combined {len(forecast_df.columns)} models")
    
    # =====================================================================
    # STEP 5: EVALUATION
    # =====================================================================
    logger.info("=" * 80)
    logger.info("STEP 5: Forecast Evaluation")
    logger.info("=" * 80)
    
    metrics_results = {}
    
    for model_name, pred_series in forecasts.items():
        logger.info(f"\nEvaluating {model_name}:")
        
        # Align with actual
        common_idx = rv_test.index.intersection(pred_series.index)
        actual = rv_test.loc[common_idx].values
        pred = pred_series.loc[common_idx].values
        
        # Calculate metrics
        returns_test_aligned = data_bundle.returns_test[primary_ticker].loc[common_idx].values
        metrics = compute_volatility_forecast_metrics(actual, pred, returns_test_aligned)
        
        metrics_results[model_name] = metrics
        
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  QLIKE: {metrics['qlike']:.4f}")
        logger.info(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
    
    # Model comparison (Diebold-Mariano test)
    if 'Ensemble' in forecasts and 'GARCH' in forecasts:
        logger.info("\nDiebold-Mariano Test (Ensemble vs GARCH):")
        common_idx = rv_test.index.intersection(forecasts['Ensemble'].index).intersection(forecasts['GARCH'].index)
        dm_result = compute_diebold_mariano_test(
            rv_test.loc[common_idx].values,
            forecasts['Ensemble'].loc[common_idx].values,
            forecasts['GARCH'].loc[common_idx].values
        )
        logger.info(f"  DM Statistic: {dm_result['dm_statistic']:.4f}")
        logger.info(f"  P-value: {dm_result['p_value']:.4f}")
        logger.info(f"  Ensemble better: {dm_result['model1_better']}")
        logger.info(f"  Significant (p<0.05): {dm_result['significant']}")
    
    # Find best model
    best_model = min(metrics_results.keys(), 
                     key=lambda k: metrics_results[k].get('rmse', float('inf')))
    logger.info(f"\nBest Model (by RMSE): {best_model}")
    
    # =====================================================================
    # STEP 6: VISUALIZATIONS
    # =====================================================================
    logger.info("=" * 80)
    logger.info("STEP 6: Visualizations")
    logger.info("=" * 80)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 14))
    
    # Plot 1: Volatility forecasts
    ax1 = plt.subplot(3, 3, 1)
    plot_volatility_forecast(rv_test, forecasts, 
                            title=f"{primary_ticker} Volatility Forecasts", ax=ax1)
    
    # Plot 2: Forecast errors
    ax2 = plt.subplot(3, 3, 2)
    plot_volatility_forecast_errors(rv_test, forecasts, ax=ax2)
    
    # Plot 3: Scatter plot (best model)
    ax3 = plt.subplot(3, 3, 3)
    common_idx = rv_test.index.intersection(forecasts[best_model].index)
    plot_volatility_scatter(
        rv_test.loc[common_idx].values,
        forecasts[best_model].loc[common_idx].values,
        model_name=best_model,
        ax=ax3
    )
    
    # Plot 4: Realized vol estimators comparison
    ax4 = plt.subplot(3, 3, 4)
    rv_estimators = {}
    for est in vol_estimators[:3]:  # Limit to 3
        col = f"{primary_ticker}_{est}"
        if col in data_bundle.realized_vol_test.columns:
            rv_estimators[est] = data_bundle.realized_vol_test[col]
    plot_realized_volatility_comparison(rv_estimators, 
                                       title=f"{primary_ticker} RV Estimators", ax=ax4)
    
    # Plot 5: Volatility cone
    ax5 = plt.subplot(3, 3, 5)
    rv_full = pd.concat([rv_train, rv_test])
    plot_volatility_cone(rv_full, ax=ax5)
    
    # Plot 6: Model comparison metrics
    ax6 = plt.subplot(3, 3, 6)
    plot_model_comparison_metrics(metrics_results, ax=ax6)
    
    # Plot 7: Volatility surface (multi-asset)
    ax7 = plt.subplot(3, 3, 7)
    if len(tickers) > 1:
        vol_matrix = np.stack([
            data_bundle.realized_vol_test[f"{t}_standard"].values 
            for t in tickers[:10]  # Limit to 10 assets
            if f"{t}_standard" in data_bundle.realized_vol_test.columns
        ])
        plot_volatility_surface(
            data_bundle.realized_vol_test.index,
            tickers[:vol_matrix.shape[0]],
            vol_matrix.T,
            title="Multi-Asset Volatility Surface",
            ax=ax7
        )
    
    # Plot 8: Time series of returns and volatility
    ax8 = plt.subplot(3, 3, 8)
    returns_test = data_bundle.returns_test[primary_ticker]
    ax8_twin = ax8.twinx()
    ax8.plot(returns_test.index, returns_test.values, 
             color='gray', alpha=0.5, label='Returns')
    ax8_twin.plot(rv_test.index, rv_test.values, 
                  color='red', linewidth=2, label='Realized Vol')
    ax8.set_xlabel('Date')
    ax8.set_ylabel('Returns', color='gray')
    ax8_twin.set_ylabel('Volatility', color='red')
    ax8.set_title(f'{primary_ticker} Returns vs Volatility')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Forecast vs actual distribution
    ax9 = plt.subplot(3, 3, 9)
    ax9.hist(rv_test.values, bins=30, alpha=0.5, label='Actual', density=True)
    for model_name in forecasts.keys():
        common_idx = rv_test.index.intersection(forecasts[model_name].index)
        ax9.hist(forecasts[model_name].loc[common_idx].values, 
                bins=30, alpha=0.3, label=model_name, density=True)
    ax9.set_xlabel('Volatility')
    ax9.set_ylabel('Density')
    ax9.set_title('Volatility Distribution')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = out / f"{run_id}_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization: {fig_path}")
    plt.close()
    
    # =====================================================================
    # STEP 7: EXPORT RESULTS
    # =====================================================================
    logger.info("=" * 80)
    logger.info("STEP 7: Export Results")
    logger.info("=" * 80)
    
    # Save metrics comparison
    metrics_df = pd.DataFrame(metrics_results).T
    metrics_path = out / f"{run_id}_metrics.csv"
    metrics_df.to_csv(metrics_path)
    logger.info(f"Saved metrics: {metrics_path}")
    
    # Save forecasts
    forecasts_df = pd.DataFrame(forecasts)
    forecasts_df['actual'] = rv_test
    forecasts_path = out / f"{run_id}_forecasts.csv"
    forecasts_df.to_csv(forecasts_path)
    logger.info(f"Saved forecasts: {forecasts_path}")
    
    # Save volatility characteristics
    char_path = out / f"{run_id}_characteristics.csv"
    char_data = []
    for ticker, chars in vol_characteristics.items():
        row = {'ticker': ticker}
        row.update(chars['persistence'])
        row.update(chars['asymmetry'])
        char_data.append(row)
    pd.DataFrame(char_data).to_csv(char_path, index=False)
    logger.info(f"Saved volatility characteristics: {char_path}")
    
    # =====================================================================
    # STEP 8: CHECKPOINT
    # =====================================================================
    state.update({
        'config': config,
        'run_id': run_id,
        'primary_ticker': primary_ticker,
        'tickers': tickers,
        'metrics': metrics_results,
        'best_model': best_model
    })
    
    checkpoint_path = out / f"{run_id}_checkpoint.pkl"
    save_checkpoint(state, str(checkpoint_path))
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Primary Ticker: {primary_ticker}")
    logger.info(f"Models Trained: {', '.join(trained_models.keys())}")
    logger.info(f"Best Model: {best_model}")
    logger.info(f"Best RMSE: {metrics_results[best_model]['rmse']:.4f}")
    logger.info(f"Best R²: {metrics_results[best_model]['r2']:.4f}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ensemble Volatility Forecasting Pipeline"
    )
    parser.add_argument(
        "--config",
        default="configs/volatility_ensemble.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output",
        default="outputs/volatility_forecast",
        help="Output directory",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    try:
        run_volatility_forecasting_pipeline(args.config, args.output, args.resume)
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
