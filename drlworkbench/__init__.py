"""
DRLWorkbench - Deep Reinforcement Learning Backtesting and Analysis Framework.

A comprehensive framework for developing, testing, and deploying Deep Reinforcement 
Learning agents for quantitative finance applications.
"""

__version__ = "0.1.0"
__author__ = "DRLWorkbench Team"
__email__ = "info@drlworkbench.io"

# Import main components for easy access
from drlworkbench.backtesting import BacktestEngine
from drlworkbench.regime import RegimeDetector
from drlworkbench.validation import DataValidator
from drlworkbench.utils import setup_logger

__all__ = [
    "BacktestEngine",
    "RegimeDetector",
    "DataValidator",
    "setup_logger",
    "__version__",
]
