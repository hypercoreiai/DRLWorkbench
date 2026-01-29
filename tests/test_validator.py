# Unit tests for data validation (V3 — PROJECT_OUTLINE Section 12)

import numpy as np
import pandas as pd
import pytest

from src.data.validator import DataValidator
from src.utils import DataValidationError


class TestDataValidator:
    def test_check_missing_data_pass(self):
        v = DataValidator()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        v.check_missing_data(df, threshold=0.05)

    def test_check_missing_data_fail(self):
        v = DataValidator()
        df = pd.DataFrame({"a": [1, np.nan, np.nan], "b": [4, 5, 6]})
        with pytest.raises(DataValidationError):
            v.check_missing_data(df, threshold=0.05)

    def test_check_outliers_iqr(self):
        v = DataValidator()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100]})
        out = v.check_outliers(df, method="iqr")
        assert out is not None
        assert out["a"].max() <= 100

    def test_check_data_leakage_fail(self):
        v = DataValidator()
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        X_train = pd.DataFrame({"x": range(5)}, index=idx[:5])
        X_test = pd.DataFrame({"x": range(5)}, index=idx[2:7])
        with pytest.raises(DataValidationError):
            v.check_data_leakage(X_train, X_test)

    def test_check_data_leakage_pass(self):
        v = DataValidator()
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        X_train = pd.DataFrame({"x": range(5)}, index=idx[:5])
        X_test = pd.DataFrame({"x": range(5)}, index=idx[5:10])
        v.check_data_leakage(X_train, X_test)

    def test_check_collinearity(self):
        v = DataValidator()
        df = pd.DataFrame({"a": range(10), "b": range(10), "c": np.random.randn(10)})
        pairs = v.check_collinearity(df, threshold=0.99)
        assert ("a", "b") in pairs or ("b", "a") in pairs
