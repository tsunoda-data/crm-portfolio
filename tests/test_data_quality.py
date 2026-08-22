"""
データ品質チェッカーのテスト
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.quality.data_quality import DataQualityChecker


@pytest.fixture
def checker():
    """ルールなしのチェッカー（個別メソッドテスト用）。"""
    return DataQualityChecker(config=None)


@pytest.fixture
def good_df():
    """品質チェックを通過する正常なDataFrame。"""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "age": np.random.randint(18, 70, n),
        "total_spend": np.random.randint(0, 50000, n),
        "order_count": np.random.poisson(5, n).astype(float),
    })


@pytest.fixture
def bad_df():
    """品質問題を含むDataFrame。"""
    n = 50
    df = pd.DataFrame({
        "customer_id": ["C00001"] * 10 + [f"C{i:05d}" for i in range(2, n - 8)],
        "age": list(range(n)),
        "total_spend": [-100.0] * n,
        "order_count": [float("nan")] * n,
    })
    return df


class TestDataQualityChecker:
    def test_instantiation(self, checker):
        assert checker is not None

    def test_check_null_rates_pass(self, checker, good_df):
        issues = checker.check_null_rates(good_df, max_null_rates={"order_count": 0.05})
        assert len(issues) == 0

    def test_check_null_rates_fail(self, checker, bad_df):
        issues = checker.check_null_rates(bad_df, max_null_rates={"order_count": 0.05})
        assert len(issues) > 0

    def test_check_duplicates_pass(self, checker, good_df):
        issues = checker.check_duplicates(good_df, key_column="customer_id")
        assert len(issues) == 0

    def test_check_duplicates_fail(self, checker, bad_df):
        issues = checker.check_duplicates(bad_df, key_column="customer_id")
        assert len(issues) > 0

    def test_check_row_count_pass(self, checker, good_df):
        issues = checker.check_row_count(good_df, min_rows=50, max_rows=200)
        assert len(issues) == 0

    def test_check_row_count_fail_too_few(self, checker, bad_df):
        issues = checker.check_row_count(bad_df, min_rows=100, max_rows=200)
        assert len(issues) > 0

    def test_check_value_ranges(self, checker, bad_df):
        issues = checker.check_value_ranges(bad_df, ranges_dict={"total_spend": {"min": 0}})
        assert len(issues) > 0

    def test_check_schema_pass(self, checker, good_df):
        issues = checker.check_schema(good_df, expected_columns=["customer_id", "age"])
        assert len(issues) == 0

    def test_check_schema_fail(self, checker, good_df):
        issues = checker.check_schema(good_df, expected_columns=["customer_id", "nonexistent_col"])
        assert len(issues) > 0
