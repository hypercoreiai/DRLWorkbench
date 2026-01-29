"""Data validation and quality checking."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import logging

from drlworkbench.utils.exceptions import DataQualityError

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate data quality and run statistical tests.
    
    Performs checks for stationarity, normality, outliers, missing data, etc.
    """
    
    def __init__(
        self,
        missing_threshold: float = 0.05,
        outlier_threshold: float = 3.0
    ):
        """
        Initialize data validator.
        
        Parameters
        ----------
        missing_threshold : float, default 0.05
            Maximum acceptable fraction of missing data (0.05 = 5%).
        outlier_threshold : float, default 3.0
            Z-score threshold for outlier detection.
        """
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        
        logger.info(
            f"DataValidator initialized with missing_threshold={missing_threshold}, "
            f"outlier_threshold={outlier_threshold}"
        )
    
    def validate_all(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Run all validation checks.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to validate.
            
        Returns
        -------
        Dict[str, any]
            Dictionary containing validation results:
            - is_valid: bool
            - missing_data: dict
            - outliers: dict
            - issues: list of strings
        """
        logger.info(f"Running validation on data with shape {data.shape}")
        
        results = {
            'is_valid': True,
            'missing_data': {},
            'outliers': {},
            'duplicates': 0,
            'issues': []
        }
        
        # Check missing data
        missing_check = self.check_missing_data(data)
        results['missing_data'] = missing_check
        if any(v > self.missing_threshold for v in missing_check.values()):
            results['is_valid'] = False
            results['issues'].append(
                f"Missing data exceeds threshold: {missing_check}"
            )
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        results['duplicates'] = duplicates
        if duplicates > 0:
            results['issues'].append(f"Found {duplicates} duplicate rows")
        
        # Check for outliers
        outlier_check = self.check_outliers(data)
        results['outliers'] = outlier_check
        
        # Check data types
        type_check = self.check_data_types(data)
        if not type_check['valid']:
            results['is_valid'] = False
            results['issues'].extend(type_check['issues'])
        
        logger.info(
            f"Validation complete. Valid: {results['is_valid']}, "
            f"Issues: {len(results['issues'])}"
        )
        
        return results
    
    def check_missing_data(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Check for missing data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to check.
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping column names to fraction of missing data.
        """
        missing = data.isnull().sum() / len(data)
        return {col: val for col, val in missing.items() if val > 0}
    
    def check_outliers(
        self,
        data: pd.DataFrame,
        method: str = "zscore"
    ) -> Dict[str, int]:
        """
        Check for outliers using Z-score method.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to check.
        method : str, default "zscore"
            Method to use ("zscore" or "iqr").
            
        Returns
        -------
        Dict[str, int]
            Dictionary mapping column names to number of outliers.
        """
        outliers = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if method == "zscore":
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                n_outliers = (z_scores > self.outlier_threshold).sum()
            elif method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                n_outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            else:
                n_outliers = 0
            
            if n_outliers > 0:
                outliers[col] = n_outliers
        
        return outliers
    
    def check_data_types(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Check if data types are appropriate.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to check.
            
        Returns
        -------
        Dict[str, any]
            Dictionary with 'valid' flag and list of 'issues'.
        """
        issues = []
        
        # Check if numeric columns contain non-numeric values
        for col in data.columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    issues.append(f"Column '{col}' should be numeric but is {data[col].dtype}")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                if (data[col] < 0).any():
                    issues.append(f"Column '{col}' contains negative values")
        
        # Check for negative volume
        if 'volume' in data.columns:
            if (data['volume'] < 0).any():
                issues.append("Volume contains negative values")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def test_stationarity(
        self,
        series: pd.Series,
        significance_level: float = 0.05
    ) -> Dict[str, any]:
        """
        Test for stationarity using Augmented Dickey-Fuller test.
        
        Parameters
        ----------
        series : pd.Series
            Time series to test.
        significance_level : float, default 0.05
            Significance level for the test.
            
        Returns
        -------
        Dict[str, any]
            Dictionary containing test results:
            - is_stationary: bool
            - test_statistic: float
            - p_value: float
            - critical_values: dict
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 12:
            logger.warning("Series too short for ADF test")
            return {
                'is_stationary': None,
                'test_statistic': None,
                'p_value': None,
                'critical_values': None,
                'error': 'Series too short'
            }
        
        result = adfuller(series_clean, autolag='AIC')
        
        return {
            'is_stationary': result[1] < significance_level,
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
        }
