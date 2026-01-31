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

import yaml

from src.utils import ConfigError
from src.utils.logging import setup_logging
from src.utils.checkpoint import save_checkpoint, load_checkpoint


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
    
    # 3. Data Pipeline
    from src.data.pipeline import DataPipeline
    data_pipe = DataPipeline(config)
    data_bundle = data_pipe.run()
    logger.info("Data pipeline completed.")

    # 4. Crypto Validation (V4)
    # Assuming data_bundle.X_train or similar holds the main DF for now
    # For this stub, we'll try to find a dataframe in the bundle to validate
    # logic below detects if X_train/X_test exists and validates them.
    from src.data.validator import DataValidator, DataValidationError
    validator = DataValidator()
    
    # Simple heuristic to get data for validation (in real app, specific field)
    # Using 'X_train' as placeholder if populated, else if bundle.metadata has data
    # For now, we'll skip if empty, but log check
    target_data = getattr(data_bundle, "X_train", None)
    
    if target_data is not None and hasattr(target_data, "index"):
        try:
            # Check crypto continuity
            validator.check_crypto_continuity(target_data)
            logger.info("Crypto continuity check passed.")
        except DataValidationError as e:
            logger.warning("Crypto continuity check failed: %s", e)
    
    # 5. Stress Testing (V4)
    # Check if stress testing is requested in config
    stress_config = config.get("stress_test", {})
    if stress_config and target_data is not None:
        from src.backtest import SyntheticGenerator
        gen = SyntheticGenerator(target_data)
        if stress_config.get("flash_crash"):
            logger.info("Injecting Flash Crash scenario...")
            target_data = gen.inject_flash_crash(**stress_config["flash_crash"])
            # Update bundle
            data_bundle.X_train = target_data

    # 6. Walk-Forward Backtest (V3)
    from src.backtest import WalkForwardBacktester
    
    # Logic: use the data bundle. If bundle is empty (stub), we might fail.
    # We will pass the bundle object or data DataFrame to walker.
    # The walker expects 'data' in constructor.
    walker = WalkForwardBacktester(target_data, config)
    
    try:
        report = walker.run()
        logger.info("Backtest completed. Steps: %d", len(report.get("regime_info", [])))
        state["backtest_report"] = report
    except Exception:
        logger.exception("Backtest failed during walk-forward backtest.")
        raise
    state["config"] = config
    state["run_id"] = run_id
    checkpoint_path = out / "checkpoint.pkl"
    save_checkpoint(state, str(checkpoint_path))
    logger.info("Checkpoint saved: %s", checkpoint_path)


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
