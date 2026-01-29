# Portfolio simulator with transaction costs and slippage (V3 — PROJECT_OUTLINE Section 4.2)

from typing import Any, Dict, Union

import numpy as np


class PortfolioSimulator:
    """
    Model realistic trading: turnover, costs, effective returns.
    """

    def rebalance(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        prices: Union[np.ndarray, Any],
        bid_ask_spread: float = 0.001,
        commission: float = 0.001,
    ) -> Dict[str, float]:
        """
        Calculate turnover, costs, effective returns.

        turnover = sum(|new_weights - old_weights|) / 2
        cost = turnover * (bid_ask_spread + commission)

        Args:
            old_weights: Previous portfolio weights.
            new_weights: Target weights.
            prices: Current prices (for value-based turnover if needed).
            bid_ask_spread: Spread as fraction.
            commission: Commission per trade as fraction.

        Returns:
            Dict with 'turnover', 'cost', and optionally 'effective_return'.
        """
        turnover = np.abs(np.asarray(new_weights) - np.asarray(old_weights)).sum() / 2.0
        cost = turnover * (bid_ask_spread + commission)
        return {"turnover": float(turnover), "cost": float(cost)}
