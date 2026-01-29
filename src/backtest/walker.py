# Walk-forward validation framework (V3 — PROJECT_OUTLINE Section 4.1)

from typing import Any, Callable, Dict, List, Optional
import logging
import numpy as np
import pandas as pd

from src.utils.errors import BacktestError
from src.backtest.simulator import PortfolioSimulator
from src.backtest.regime import detect_regimes, compute_regime_metrics

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    For each test period, retrain model and rebalance portfolio.
    """

    def __init__(self, data: Any, config: Dict[str, Any]) -> None:
        """
        Args:
            data: Full OHLCV / feature data (DataBundle or DataFrame).
            config: train_window, test_window, rebalance_freq.
        """
        self.data = data
        self.config = config
        self.train_window = config.get("backtest", {}).get("train_window", 252)
        self.test_window = config.get("backtest", {}).get("test_window", 63)
        self.rebalance_freq = config.get("backtest", {}).get(
            "rebalance_freq", 21
        )
        self.simulator = PortfolioSimulator()

    def run(
        self,
        model_builder: Optional[Callable] = None,
        optimizer_builder: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Loop over time windows:
        1. [t0, t0+train_window): Train model & optimizer.
        2. [t0+train_window, t0+train_window+test_window): Test & trade.
        3. Record PnL, weights, predictions.
        4. Slide window forward by rebalance_freq.

        Returns:
            Backtest report with cumulative returns, metrics, regime info.
        """
        if model_builder is None:
            raise BacktestError("model_builder is required")
        
        # Extract data from DataBundle if needed
        if hasattr(self.data, 'X_train'):
            # Using DataBundle - combine train and test for walk-forward
            X_all = np.vstack([self.data.X_train, self.data.X_test])
            y_all = np.concatenate([self.data.y_train, self.data.y_test])
        else:
            # Assume data is already prepared
            X_all = self.data
            y_all = None
        
        n_samples = len(X_all)
        min_train_size = self.train_window
        
        if n_samples < min_train_size + self.test_window:
            raise BacktestError(
                f"Insufficient data: need at least {min_train_size + self.test_window} samples, "
                f"got {n_samples}"
            )
        
        # Storage for results
        all_predictions = []
        all_actuals = []
        all_returns = []
        all_regimes = []
        timestamps = []
        
        # Walk-forward loop
        start_idx = 0
        step = 0
        
        while start_idx + min_train_size + self.test_window <= n_samples:
            step += 1
            train_end = start_idx + min_train_size
            test_end = min(train_end + self.test_window, n_samples)
            
            logger.info(
                f"Step {step}: Train [{start_idx}:{train_end}], "
                f"Test [{train_end}:{test_end}]"
            )
            
            # Split data
            X_train = X_all[start_idx:train_end]
            X_test = X_all[train_end:test_end]
            
            if y_all is not None:
                y_train = y_all[start_idx:train_end]
                y_test = y_all[train_end:test_end]
            else:
                # If no targets, use last feature as target (simplified)
                y_train = X_train[:, -1, 0]
                y_test = X_test[:, -1, 0]
            
            # Train model
            try:
                model = model_builder()
                config_models = self.config.get('models', [{}])
                model_config = config_models[0] if config_models else {}
                
                # Build model
                if hasattr(model, 'build'):
                    model.build(
                        input_shape=(X_train.shape[1], X_train.shape[2]),
                        config=model_config
                    )
                
                # Fit model using its fit method (which handles flattening for sklearn models)
                if hasattr(model, 'fit'):
                    trained_model, history = model.fit(
                        X_train, y_train,
                        X_val=None,  # Could add validation split
                        y_val=None,
                        config=model_config
                    )
                else:
                    trained_model = model
                
                # Predict using model's predict method (which handles flattening)
                if hasattr(model, 'predict'):
                    predictions = model.predict(X_test)
                else:
                    predictions = np.zeros(len(X_test))
                
                # Store results
                all_predictions.extend(predictions.tolist())
                all_actuals.extend(y_test.tolist())
                
                # Compute returns (simplified: just use predictions as signals)
                returns = np.sign(predictions) * y_test
                all_returns.extend(returns.tolist())
                
                timestamps.extend(range(train_end, test_end))
                
            except Exception as e:
                logger.warning(f"Step {step} failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # Continue with zeros
                all_predictions.extend([0.0] * len(X_test))
                all_actuals.extend(y_test.tolist() if y_all is not None else [0.0] * len(X_test))
                all_returns.extend([0.0] * len(X_test))
                timestamps.extend(range(train_end, test_end))
            
            # Slide window
            start_idx += self.rebalance_freq
        
        # Convert to series for analysis
        returns_series = pd.Series(all_returns)
        
        # Detect regimes
        regime_config = self.config.get("backtest", {}).get("regime_detection", {})
        regimes = detect_regimes(
            returns_series,
            method=regime_config.get("method", "volatility"),
            window=regime_config.get("periods", 63)
        )
        
        # Compute metrics
        cumulative_returns = (1 + returns_series).cumprod()
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
        max_dd = self._compute_max_drawdown(cumulative_returns.values)
        
        # Regime-specific metrics
        regime_metrics = compute_regime_metrics(returns_series, regimes)
        
        report: Dict[str, Any] = {
            "cumulative_returns": cumulative_returns.tolist(),
            "returns": all_returns,
            "predictions": all_predictions,
            "actuals": all_actuals,
            "metrics": {
                "sharpe": float(sharpe),
                "max_drawdown": float(max_dd),
                "total_return": float(cumulative_returns.iloc[-1] - 1) if len(cumulative_returns) > 0 else 0,
                "n_steps": step,
            },
            "regime_info": regime_metrics.to_dict('records') if not regime_metrics.empty else [],
            "weights_history": [],  # Placeholder for portfolio weights
        }
        
        logger.info(f"Backtest complete: Sharpe={sharpe:.3f}, MaxDD={max_dd:.3f}")
        return report
    
    def _compute_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Compute maximum drawdown."""
        if len(cumulative_returns) == 0:
            return 0.0
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return float(np.abs(drawdown.min())) if len(drawdown) > 0 else 0.0
