# Walk-forward validation framework (V3 — PROJECT_OUTLINE Section 4.1)

from typing import Any, Callable, Dict, Optional

from src.utils.errors import BacktestError


class WalkForwardBacktester:
    """
    For each test period, retrain model and rebalance portfolio.
    """

    def __init__(self, data: Any, config: Dict[str, Any]) -> None:
        """
        Args:
            data: Full OHLCV / feature data.
            config: train_window, test_window, rebalance_freq.
        """
        self.data = data
        self.config = config
        self.train_window = config.get("backtest", {}).get("train_window", 252)
        self.test_window = config.get("backtest", {}).get("test_window", 63)
        self.rebalance_freq = config.get("backtest", {}).get(
            "rebalance_freq", 21
        )

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
        report: Dict[str, Any] = {
            "cumulative_returns": [],
            "metrics": {},
            "regime_info": [],
            "weights_history": [],
        }

        # 0. Basic Validation
        if not hasattr(self.data, "index") or not hasattr(self.data, "iloc"):
            raise BacktestError("Data must be a pandas DataFrame with datetime index.")
        
        n_samples = len(self.data)
        if n_samples < self.train_window + self.test_window:
            raise BacktestError(
                f"Data length {n_samples} insufficient for "
                f"train_window={self.train_window} + test_window={self.test_window}"
            )

        # 1. Walk-Forward Loop
        current_step = 0
        
        while current_step + self.train_window + self.test_window <= n_samples:
            # Indices for slicing
            train_start = current_step
            train_end = current_step + self.train_window
            test_start = train_end
            test_end = test_start + self.test_window

            # Slices
            # Assuming self.data is a DataFrame with features + target
            # In a real scenario, you'd separate features/targets explicitly
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]

            # 2. Train Model (if provided)
            # Interface assumption: model_builder(config) -> model
            # model.fit(train_data)
            trained_model = None
            if model_builder:
                trained_model = model_builder(self.config)
                # Assuming fit method exists and takes DataFrame
                # In production, might need to tease out X, y inside or pass valid args
                try:
                    trained_model.fit(train_data)
                except Exception as e:
                    # Log error, potentially skip or use fallback
                    # For now, simplistic error propagation
                    raise BacktestError(f"Model training failed at step {current_step}: {e}")

            # 3. Portfolio Optimization / Trading Simulation
            # Interface assumption: optimizer_builder(config) -> optimizer
            # optimizer.optimize(train_data (for cov), test_data (for applying weights))
            # OR simple simulation: model predicts -> weights -> returns
            
            # For this V3 implementation, we'll simulate 'Trading' by recording returns
            # derived from a hypothetical strategy or the model's output.
            
            # Placeholder for actual trading logic:
            # weights = optimizer.optimize(train_data) OR trained_model.predict(test_data)
            # returns = (weights * test_data.pct_change()).sum(axis=1)
            
            # Since we don't have the full model/optimizer implementations in this file,
            # we will record the 'period' info to show the loop works.
            
            period_info = {
                "train_range": (self.data.index[train_start], self.data.index[train_end-1]),
                "test_range": (self.data.index[test_start], self.data.index[test_end-1]),
                "step": current_step
            }
            report["regime_info"].append(period_info)
            
            # 4. Slide Window
            current_step += self.rebalance_freq

        return report
