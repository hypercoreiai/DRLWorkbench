"""Tests for data validation module."""

import pytest
import pandas as pd
import numpy as np

from drlworkbench.validation import DataValidator


def test_data_validator_initialization():
    """Test DataValidator initialization."""
    validator = DataValidator(missing_threshold=0.1, outlier_threshold=3.5)
    
    assert validator.missing_threshold == 0.1
    assert validator.outlier_threshold == 3.5


def test_check_missing_data():
    """Test missing data check."""
    validator = DataValidator()
    
    data = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
        'b': [1, 2, 3, 4],
        'c': [np.nan, np.nan, 3, 4]
    })
    
    missing = validator.check_missing_data(data)
    
    assert 'a' in missing
    assert 'c' in missing
    assert 'b' not in missing
    assert missing['a'] == 0.25
    assert missing['c'] == 0.5


def test_check_outliers_zscore():
    """Test outlier detection with Z-score."""
    validator = DataValidator(outlier_threshold=2.0)
    
    np.random.seed(42)
    data = pd.DataFrame({
        'normal': np.random.normal(0, 1, 100),
        'with_outliers': np.concatenate([
            np.random.normal(0, 1, 95),
            [10, 15, -12, 20, -18]  # Clear outliers
        ])
    })
    
    outliers = validator.check_outliers(data, method="zscore")
    
    assert 'with_outliers' in outliers
    assert outliers['with_outliers'] > 0


def test_check_data_types():
    """Test data type checking."""
    validator = DataValidator()
    
    # Valid data
    valid_data = pd.DataFrame({
        'open': [100.0, 101.0, 102.0],
        'close': [101.0, 102.0, 103.0],
        'volume': [1000, 1100, 1200]
    })
    
    result = validator.check_data_types(valid_data)
    assert result['valid']
    
    # Invalid data (negative prices)
    invalid_data = pd.DataFrame({
        'open': [100.0, -101.0, 102.0],
        'close': [101.0, 102.0, 103.0],
        'volume': [1000, 1100, -1200]
    })
    
    result = validator.check_data_types(invalid_data)
    assert not result['valid']
    assert len(result['issues']) > 0


def test_validate_all():
    """Test comprehensive validation."""
    validator = DataValidator()
    
    data = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0],
        'high': [101.0, 102.0, 103.0, 104.0],
        'low': [99.0, 100.0, 101.0, 102.0],
        'close': [100.5, 101.5, 102.5, 103.5],
        'volume': [1000, 1100, 1200, 1300]
    })
    
    results = validator.validate_all(data)
    
    assert 'is_valid' in results
    assert 'missing_data' in results
    assert 'outliers' in results
    assert 'duplicates' in results
    assert 'issues' in results


def test_stationarity_test():
    """Test stationarity testing."""
    validator = DataValidator()
    
    # Non-stationary series (random walk)
    np.random.seed(42)
    non_stationary = pd.Series(np.cumsum(np.random.normal(0, 1, 100)))
    
    result = validator.test_stationarity(non_stationary)
    
    assert 'is_stationary' in result
    assert 'p_value' in result
    assert 'test_statistic' in result
