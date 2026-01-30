"""
Ensemble Portfolio Optimization Pipeline
Combines multiple forecasting models with portfolio optimization methods
to produce comprehensive backtesting results and metrics.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils import ConfigError
from src.data.portfolio_pipeline import PortfolioPipeline, load_index_data
from src.models.portfolio_optimization import get_optimizer, PortfolioWeights
from src.models.forecasting import LSTMForecaster, SimpleEnsembleForecaster, create_sequences
from src.analysis.risk import compute_risk_metrics
from src.analysis.forecasting import compute_forecasting_metrics
from src.analysis.summary import compute_summary_stats
from src.backtest.regime import RegimeDetector
from src.backtest.simulator import PortfolioSimulator
from src.display.plots import (
    plot_equity_curve_with_regimes,
    plot_drawdown_analysis,
    plot_rolling_metrics,
)
from src.display.export import export_results


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


def run_ensemble_portfolio_pipeline(
    config_path: str,
    output_dir: str,
    resume: Optional[str] = None,
) -> None:
    """
    Main pipeline for ensemble portfolio optimization.
    
    Steps:
    1. Load data (multi-asset OHLCV)
    2. Train ensemble forecasting models
    3. Apply portfolio optimization methods
    4. Backtest with walk-forward validation
    5. Analyze results (risk metrics, regime analysis)
    6. Generate visualizations and reports
    
    Args:
        config_path: Path to configuration YAML file.
        output_dir: Directory for outputs.
        resume: Optional checkpoint path to resume from.
    """
    # Load configuration
    config = load_config(config_path)
    run_id = config.get("display", {}).get("run_id", "ensemble_po")
    
    # Setup output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(str(log_dir), run_id)
    logger.info(f"Starting Ensemble Portfolio Optimization Pipeline")
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
    logger.info("=" * 60)
    logger.info("STEP 1: Data Pipeline")
    logger.info("=" * 60)
    
    pipeline = PortfolioPipeline(config)
    data_bundle = pipeline.run()
    
    if data_bundle.returns_train is None or data_bundle.returns_train.empty:
        logger.error("Data pipeline failed to produce training data")
        return
    
    logger.info(f"Loaded {len(data_bundle.metadata.get('tickers', []))} assets")
    logger.info(f"Training samples: {len(data_bundle.returns_train)}")
    logger.info(f"Testing samples: {len(data_bundle.returns_test)}")
    
    tickers = data_bundle.metadata['tickers']
    
    # Load benchmark indexes
    benchmark_tickers = config.get('data', {}).get('benchmark_tickers', ['SPY'])
    benchmarks = load_index_data(benchmark_tickers, config.get('data', {}).get('period', '2y'))
    logger.info(f"Loaded {len(benchmark_tickers)} benchmark indexes")
    
    # =====================================================================
    # STEP 2: PORTFOLIO OPTIMIZATION (Training Period)
    # =====================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Portfolio Optimization")
    logger.info("=" * 60)
    
    optimization_methods = config.get('optimization', {}).get('methods', ['risk_parity', 'omega', 'cvar'])
    optimization_params = config.get('optimization', {}).get('params', {})
    
    portfolio_weights = {}
    
    for method in optimization_methods:
        logger.info(f"Optimizing with method: {method}")
        
        try:
            # Get optimizer
            params = optimization_params.get(method, {})
            optimizer = get_optimizer(method, params)
            
            # Optimize on training returns
            weights = optimizer.optimize(data_bundle.returns_train)
            portfolio_weights[method] = weights
            
            logger.info(f"  {method} weights: {weights.weights}")
            logger.info(f"  Metadata: {weights.metadata}")
            
        except Exception as e:
            logger.error(f"  Failed to optimize with {method}: {e}")
            continue
    
    if not portfolio_weights:
        logger.error("No portfolio optimization methods succeeded")
        return
    
    # =====================================================================
    # STEP 3: BACKTESTING WITH PORTFOLIO SIMULATOR
    # =====================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Backtesting")
    logger.info("=" * 60)
    
    simulator = PortfolioSimulator()
    transaction_cost = config.get('backtest', {}).get('transaction_cost', 0.001)
    bid_ask_spread = config.get('backtest', {}).get('bid_ask_spread', 0.001)
    
    backtest_results = {}
    
    for method, weights in portfolio_weights.items():
        logger.info(f"Backtesting strategy: {method}")
        
        # Get weights as array
        w = weights.to_array(tickers)
        
        # Calculate portfolio returns (test period)
        returns_test = data_bundle.returns_test
        portfolio_returns = (returns_test * w).sum(axis=1)
        
        # Simulate rebalancing costs
        # Assume rebalance at start of test period
        costs = simulator.rebalance(
            old_weights=np.ones(len(tickers)) / len(tickers),  # Equal weight start
            new_weights=w,
            prices=data_bundle.prices_test.iloc[0].values,
            bid_ask_spread=bid_ask_spread,
            commission=transaction_cost
        )
        
        # Adjust first return for transaction costs
        portfolio_returns.iloc[0] -= costs['cost']
        
        backtest_results[method] = {
            'returns': portfolio_returns,
            'weights': weights,
            'costs': costs
        }
        
        logger.info(f"  Total return: {portfolio_returns.sum():.4f}")
        logger.info(f"  Transaction cost: {costs['cost']:.4f}")
    
    # =====================================================================
    # STEP 4: REGIME DETECTION
    # =====================================================================
    logger.info("=" * 60)
    logger.info("STEP 4: Regime Detection")
    logger.info("=" * 60)
    
    regime_method = config.get('backtest', {}).get('regime_detection', {}).get('method', 'volatility')
    n_regimes = config.get('backtest', {}).get('regime_detection', {}).get('n_regimes', 3)
    
    detector = RegimeDetector(method=regime_method, n_regimes=n_regimes)
    
    # Detect regimes on combined returns
    combined_returns = data_bundle.returns_test.mean(axis=1)
    regimes = detector.fit_predict(combined_returns)
    
    logger.info(f"Detected regimes: {np.unique(regimes)}")
    logger.info(f"Regime distribution: {pd.Series(regimes).value_counts().to_dict()}")
    
    # =====================================================================
    # STEP 5: RISK METRICS ANALYSIS
    # =====================================================================
    logger.info("=" * 60)
    logger.info("STEP 5: Risk Metrics Analysis")
    logger.info("=" * 60)
    
    metrics_results = {}
    
    for method, result in backtest_results.items():
        logger.info(f"Computing metrics for: {method}")
        
        returns = result['returns']
        
        # Overall metrics
        metrics = compute_risk_metrics(returns, regime_labels=regimes)
        metrics_results[method] = metrics
        
        logger.info(f"  Sharpe Ratio: {metrics['sharpe'].iloc[0]:.3f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown'].iloc[0]:.3f}")
        logger.info(f"  Annual Return: {metrics['annual_return'].iloc[0]:.3f}")
        logger.info(f"  Volatility: {metrics['volatility'].iloc[0]:.3f}")
    
    # =====================================================================
    # STEP 6: COMPARATIVE ANALYSIS
    # =====================================================================
    logger.info("=" * 60)
    logger.info("STEP 6: Comparative Analysis")
    logger.info("=" * 60)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        method: {
            'sharpe': metrics_results[method]['sharpe'].iloc[0],
            'sortino': metrics_results[method]['sortino'].iloc[0],
            'max_drawdown': metrics_results[method]['max_drawdown'].iloc[0],
            'calmar': metrics_results[method]['calmar'].iloc[0],
            'annual_return': metrics_results[method]['annual_return'].iloc[0],
            'volatility': metrics_results[method]['volatility'].iloc[0],
        }
        for method in backtest_results.keys()
    }).T
    
    logger.info("Comparative Results:")
    logger.info("\n" + comparison_df.to_string())
    
    # Find best strategy
    best_method = comparison_df['sharpe'].idxmax()
    logger.info(f"\nBest Strategy (by Sharpe): {best_method}")
    
    # =====================================================================
    # STEP 7: VISUALIZATIONS
    # =====================================================================
    logger.info("=" * 60)
    logger.info("STEP 7: Visualizations")
    logger.info("=" * 60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Equity curves
    ax1 = plt.subplot(3, 2, 1)
    strategies_returns = {method: result['returns'] for method, result in backtest_results.items()}
    plot_equity_curve_with_regimes(
        returns=backtest_results[best_method]['returns'],
        regime_labels=regimes,
        strategies=strategies_returns,
        ax=ax1
    )
    ax1.set_title("Equity Curves Comparison")
    
    # Plot 2: Drawdown analysis
    ax2 = plt.subplot(3, 2, 2)
    plot_drawdown_analysis(
        returns=backtest_results[best_method]['returns'],
        strategies=strategies_returns,
        ax=ax2
    )
    ax2.set_title(f"Drawdown Analysis ({best_method})")
    
    # Plot 3: Rolling Sharpe
    ax3 = plt.subplot(3, 2, 3)
    plot_rolling_metrics(
        returns=backtest_results[best_method]['returns'],
        window=60,
        metrics=['sharpe'],
        ax=ax3
    )
    ax3.set_title("Rolling Sharpe Ratio")
    
    # Plot 4: Rolling Volatility
    ax4 = plt.subplot(3, 2, 4)
    plot_rolling_metrics(
        returns=backtest_results[best_method]['returns'],
        window=60,
        metrics=['volatility'],
        ax=ax4
    )
    ax4.set_title("Rolling Volatility")
    
    # Plot 5: Portfolio weights
    ax5 = plt.subplot(3, 2, 5)
    weights_df = pd.DataFrame({
        method: result['weights'].weights
        for method, result in backtest_results.items()
    })
    weights_df.plot(kind='bar', ax=ax5)
    ax5.set_title("Portfolio Weights by Method")
    ax5.set_ylabel("Weight")
    ax5.legend(loc='best')
    
    # Plot 6: Metrics comparison
    ax6 = plt.subplot(3, 2, 6)
    comparison_df[['sharpe', 'sortino', 'calmar']].plot(kind='bar', ax=ax6)
    ax6.set_title("Risk-Adjusted Return Metrics")
    ax6.set_ylabel("Ratio")
    ax6.legend(loc='best')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = out / f"{run_id}_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization: {fig_path}")
    
    # =====================================================================
    # STEP 8: EXPORT RESULTS
    # =====================================================================
    logger.info("=" * 60)
    logger.info("STEP 8: Export Results")
    logger.info("=" * 60)
    
    # Save comparison table
    comparison_path = out / f"{run_id}_comparison.csv"
    comparison_df.to_csv(comparison_path)
    logger.info(f"Saved comparison table: {comparison_path}")
    
    # Save portfolio weights
    weights_path = out / f"{run_id}_weights.csv"
    weights_df.to_csv(weights_path)
    logger.info(f"Saved portfolio weights: {weights_path}")
    
    # Save returns
    for method, result in backtest_results.items():
        returns_path = out / f"{run_id}_{method}_returns.csv"
        result['returns'].to_csv(returns_path)
        logger.info(f"Saved {method} returns: {returns_path}")
    
    # =====================================================================
    # STEP 9: CHECKPOINT
    # =====================================================================
    state.update({
        'config': config,
        'run_id': run_id,
        'tickers': tickers,
        'backtest_results': {k: {'returns': v['returns'].to_dict()} for k, v in backtest_results.items()},
        'metrics': {k: v.to_dict() for k, v in metrics_results.items()},
        'comparison': comparison_df.to_dict(),
        'best_method': best_method
    })
    
    checkpoint_path = out / f"{run_id}_checkpoint.pkl"
    save_checkpoint(state, str(checkpoint_path))
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Best Strategy: {best_method}")
    logger.info(f"Best Sharpe Ratio: {comparison_df.loc[best_method, 'sharpe']:.3f}")
    logger.info(f"Best Annual Return: {comparison_df.loc[best_method, 'annual_return']:.3f}")
    logger.info(f"Best Max Drawdown: {comparison_df.loc[best_method, 'max_drawdown']:.3f}")
    logger.info(f"Results saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ensemble Portfolio Optimization Pipeline"
    )
    parser.add_argument(
        "--config",
        default="configs/portfolio_ensemble.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output",
        default="outputs/ensemble_po",
        help="Output directory",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    try:
        run_ensemble_portfolio_pipeline(args.config, args.output, args.resume)
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
