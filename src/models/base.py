# Base model interfaces with error handling (V3 — PROJECT_OUTLINE Section 3.1)

from typing import Any, Dict, Optional, Tuple

from src.utils import PredictionError


class BaseForecaster:
    """
    Base interface for forecasting models.
    """

    def build(self, input_shape: Any, config: Dict[str, Any]) -> None:
        """Build model architecture."""
        pass

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Any]:
        """Returns (model, history) with error handling."""
        return None, {}

    def predict(self, X_test: Any) -> Any:
        """Returns predictions; raises PredictionError if model not fitted."""
        raise PredictionError("Model not fitted")

    def get_hyperparams(self) -> Dict[str, Any]:
        """Return fitted hyperparameters for logging."""
        return {}


class BaseDRLAgent:
    """
    Base interface for DRL agents.
    """

    def build(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict[str, Any],
    ) -> None:
        """Build agent network."""
        pass

    def train(self, env: Any, config: Dict[str, Any]) -> Tuple[Any, Any]:
        """Returns (agent, training_history) with early stopping."""
        return None, {}

    def act(self, state: Any) -> Any:
        """Deterministic action."""
        raise NotImplementedError

    def act_stochastic(self, state: Any) -> Any:
        """Exploratory action."""
        raise NotImplementedError

    def evaluate(self, env: Any, episodes: int = 10) -> Dict[str, float]:
        """Compute test performance without learning."""
        return {}
