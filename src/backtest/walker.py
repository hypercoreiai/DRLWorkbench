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
        return report
