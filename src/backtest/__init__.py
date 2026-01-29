# Backtesting: walk-forward, simulator, regime, attribution (V3)

from .walker import WalkForwardBacktester
from .simulator import PortfolioSimulator
from .regime import detect_regimes, compute_regime_metrics
from .attribution import PerformanceAttribution
from .synthetic import SyntheticGenerator

__all__ = [
    "WalkForwardBacktester",
    "PortfolioSimulator",
    "detect_regimes",
    "compute_regime_metrics",
    "PerformanceAttribution",
    "SyntheticGenerator",
]
