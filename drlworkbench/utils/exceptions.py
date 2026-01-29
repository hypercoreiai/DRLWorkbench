"""Custom exceptions for DRLWorkbench."""


class DRLWorkbenchError(Exception):
    """Base exception for all DRLWorkbench errors."""
    pass


class BacktestError(DRLWorkbenchError):
    """Exception raised for backtest-related errors."""
    pass


class DataError(DRLWorkbenchError):
    """Exception raised for data-related errors."""
    pass


class DataQualityError(DataError):
    """Exception raised when data quality checks fail."""
    pass


class DataLeakageError(DataError):
    """Exception raised when data leakage is detected."""
    pass


class ValidationError(DRLWorkbenchError):
    """Exception raised for validation errors."""
    pass


class RegimeError(DRLWorkbenchError):
    """Exception raised for regime detection errors."""
    pass


class ModelError(DRLWorkbenchError):
    """Exception raised for model-related errors."""
    pass


class ConfigurationError(DRLWorkbenchError):
    """Exception raised for configuration errors."""
    pass


class OptimizationError(DRLWorkbenchError):
    """Exception raised for hyperparameter optimization errors."""
    pass


class ReportingError(DRLWorkbenchError):
    """Exception raised for reporting errors."""
    pass


class VisualizationError(DRLWorkbenchError):
    """Exception raised for visualization errors."""
    pass
