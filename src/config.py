"""
Central configuration for hardcoded parameters (V4 - PROJECT_OUTLINE Section 7.1).
"""

# -----------------------------------------------------------------------------
# Kraken OHLCV Settings (src/ohlcv/kraken_ohlcv.py)
# -----------------------------------------------------------------------------
KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
# Interval in minutes: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
KRAKEN_DEFAULT_INTERVAL = 60  # 1 hour
KRAKEN_MAX_BARS = 720
KRAKEN_MAX_RETRIES = 3

# Rate limiting (seconds)
KRAKEN_BASE_DELAY = 3.0       # Conservative delay (Kraken allows ~1 req/sec but strictly enforces bursts)
KRAKEN_RATE_LIMIT_DELAY = 30.0 # Aggressive backoff when rate limited
KRAKEN_TIMEOUT = 2            # Request timeout in seconds

# -----------------------------------------------------------------------------
# Data Validation Settings (src/data/validator.py)
# -----------------------------------------------------------------------------
VALIDATION_MISSING_THRESHOLD = 0.05     # Max fraction of NaNs allowed
VALIDATION_COLLINEARITY_THRESHOLD = 0.9 # Correlation threshold for warning
VALIDATION_ZSCORE_THRESHOLD = 3         # Z-score outlier threshold

# -----------------------------------------------------------------------------
# Stress Test Defaults (src/backtest/synthetic.py)
# -----------------------------------------------------------------------------
STRESS_FLASH_CRASH_MAGNITUDE = 0.20 # 20% drop
STRESS_FLASH_CRASH_DURATION = 5     # Periods
STRESS_VOLATILITY_FACTOR = 2.0      # Multiplier for returns std dev
