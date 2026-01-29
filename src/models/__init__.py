# Models: ml_dl, drl, optim, ensemble (V3)

from .base import BaseForecaster, BaseDRLAgent
from .tuning import fit_with_tuning
from .ensemble import ensemble_predict
from .sklearn_models import SklearnForecaster

# Try to import LSTM if PyTorch is available
try:
    from .lstm import LSTMForecaster, LSTMModel
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

__all__ = [
    "BaseForecaster",
    "BaseDRLAgent",
    "fit_with_tuning",
    "ensemble_predict",
    "SklearnForecaster",
]

if LSTM_AVAILABLE:
    __all__.extend(["LSTMForecaster", "LSTMModel"])
