# Main orchestrator: run_pipeline(config_path, output_dir) (V3 — PROJECT_OUTLINE Section 7.2)

"""
1. Load & validate config.
2. Setup logging to output_dir/logs/.
3. Get data (with validation).
4. Run walk-forward backtest (train, rebalance, simulate, record).
5. Analyze (comparative, regime-aware).
6. Display & export.

Checkpoint after each major step for resumability.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional
import json

import yaml

from src.utils import ConfigError
from src.utils.logging import setup_logging
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.data.api import get_data
from src.backtest.walker import WalkForwardBacktester
from src.models import SklearnForecaster


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


def run_pipeline(
    config_path: str,
    output_dir: str,
    resume: Optional[str] = None,
) -> None:
    """
    1. Load & validate config.
    2. Setup logging to output_dir/logs/.
    3. Get data (with validation).
    4. Run walk-forward backtest.
    5. Analyze.
    6. Display & export.

    Checkpoint after each major step for resumability.
    """
    config = load_config(config_path)
    run_id = config.get("display", {}).get("run_id", "run")
    out = Path(output_dir)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_dir), run_id)

    state: Dict[str, Any] = {}
    if resume:
        try:
            state = load_checkpoint(resume)
            logger.info("Resumed from checkpoint: %s", resume)
        except FileNotFoundError:
            logger.warning("Checkpoint not found, starting fresh: %s", resume)

    logger.info("Pipeline started with config: %s", config_path)
    
    try:
        # Step 1: Get data
        if 'data_bundle' not in state:
            logger.info("Step 1: Loading and preparing data...")
            data_bundle = get_data(config, validate=True, log_path=str(log_dir))
            state['data_bundle'] = data_bundle
            logger.info(
                f"Data prepared: {data_bundle.X_train.shape[0] if data_bundle.X_train is not None else 0} train samples, "
                f"{data_bundle.X_test.shape[0] if data_bundle.X_test is not None else 0} test samples"
            )
            
            # Checkpoint after data
            checkpoint_path = out / "checkpoint_data.pkl"
            save_checkpoint(state, str(checkpoint_path))
            logger.info("Data checkpoint saved: %s", checkpoint_path)
        else:
            logger.info("Loaded data from checkpoint")
            data_bundle = state['data_bundle']
        
        # Step 2: Run backtest
        if 'backtest_report' not in state:
            logger.info("Step 2: Running walk-forward backtest...")
            
            # Get model configuration
            models_config = config.get('models', [])
            if not models_config:
                logger.warning("No models specified in config, using default Ridge")
                models_config = [{'type': 'ridge', 'alpha': 1.0}]
            
            model_config = models_config[0]
            model_type = model_config.get('type', 'ridge')
            
            # Create model builder
            def build_model():
                return SklearnForecaster(model_type=model_type)
            
            # Create and run backtester
            backtester = WalkForwardBacktester(data_bundle, config)
            backtest_report = backtester.run(model_builder=build_model)
            
            state['backtest_report'] = backtest_report
            logger.info(
                f"Backtest complete: {backtest_report['metrics']['n_steps']} steps, "
                f"Sharpe={backtest_report['metrics']['sharpe']:.3f}, "
                f"MaxDD={backtest_report['metrics']['max_drawdown']:.3f}"
            )
            
            # Checkpoint after backtest
            checkpoint_path = out / "checkpoint_backtest.pkl"
            save_checkpoint(state, str(checkpoint_path))
            logger.info("Backtest checkpoint saved: %s", checkpoint_path)
        else:
            logger.info("Loaded backtest results from checkpoint")
            backtest_report = state['backtest_report']
        
        # Step 3: Export results
        logger.info("Step 3: Exporting results...")
        results_dir = out / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Export backtest report as JSON
        report_path = results_dir / f"{run_id}_report.json"
        with open(report_path, 'w') as f:
            json.dump(backtest_report, f, indent=2)
        logger.info(f"Report saved: {report_path}")
        
        # Export summary metrics
        summary_path = results_dir / f"{run_id}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Backtest Summary - {run_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of steps: {backtest_report['metrics']['n_steps']}\n")
            f.write(f"Total return: {backtest_report['metrics']['total_return']:.4f}\n")
            f.write(f"Sharpe ratio: {backtest_report['metrics']['sharpe']:.4f}\n")
            f.write(f"Max drawdown: {backtest_report['metrics']['max_drawdown']:.4f}\n")
            f.write(f"\nReturns: {len(backtest_report['returns'])} samples\n")
            f.write(f"Predictions: {len(backtest_report['predictions'])} samples\n")
            
            if backtest_report['regime_info']:
                f.write(f"\nRegime Analysis ({len(backtest_report['regime_info'])} regimes):\n")
                for regime in backtest_report['regime_info']:
                    f.write(
                        f"  Regime {regime['regime']}: "
                        f"Sharpe={regime.get('sharpe', 'N/A'):.3f}, "
                        f"MaxDD={regime.get('max_drawdown', 'N/A'):.3f}\n"
                    )
        logger.info(f"Summary saved: {summary_path}")
        
        # Final state
        state["config"] = config
        state["run_id"] = run_id
        state["status"] = "completed"
        
        checkpoint_path = out / "checkpoint_final.pkl"
        save_checkpoint(state, str(checkpoint_path))
        logger.info("Final checkpoint saved: %s", checkpoint_path)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML-DRL pipeline (V3)")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    run_pipeline(args.config, args.output, args.resume)


if __name__ == "__main__":
    main()
