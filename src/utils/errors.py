# Custom exceptions (V3 — PROJECT_OUTLINE Section 9.1)


class DataValidationError(Exception):
    """Raised when data validation fails (missing data, stationarity, leakage, etc.)."""
    pass


class APIError(Exception):
    """Raised when an external API call fails."""
    pass


class PredictionError(Exception):
    """Raised when prediction fails (e.g. model not fitted)."""
    pass


class BacktestError(Exception):
    """Raised when backtest execution fails."""
    pass


class ConfigError(Exception):
    """Raised when config validation or loading fails."""
    pass
