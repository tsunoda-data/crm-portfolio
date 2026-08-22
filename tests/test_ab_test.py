"""
A/Bテストフレームワークのテスト
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.campaigns.ab_test import (
    assign_groups,
    calculate_sample_size,
    measure_uplift,
    correct_multiple_testing,
)


@pytest.fixture
def customer_df():
    """テスト用顧客DataFrame。"""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "segment_label": np.random.choice(["Seg1", "Seg3", "Seg8"], n),
    })


class TestAssignGroups:
    def test_all_customers_assigned(self, customer_df):
        result = assign_groups(customer_df, control_ratio=0.2)
        assert "experiment_group" in result.columns
        assert result["experiment_group"].isna().sum() == 0

    def test_control_ratio_approximate(self, customer_df):
        result = assign_groups(customer_df, control_ratio=0.2)
        control_pct = (result["experiment_group"] == "Control").mean()
        # ハッシュベースなので完全に20%にはならないが、±5%以内
        assert 0.15 <= control_pct <= 0.25

    def test_deterministic(self, customer_df):
        r1 = assign_groups(customer_df, control_ratio=0.2, salt="test_v1")
        r2 = assign_groups(customer_df, control_ratio=0.2, salt="test_v1")
        assert (r1["experiment_group"] == r2["experiment_group"]).all()

    def test_different_salt_different_groups(self, customer_df):
        r1 = assign_groups(customer_df, control_ratio=0.2, salt="v1")
        r2 = assign_groups(customer_df, control_ratio=0.2, salt="v2")
        # 全く同じ割当にはならないはず
        assert not (r1["experiment_group"] == r2["experiment_group"]).all()


class TestCalculateSampleSize:
    def test_returns_positive_int(self):
        n = calculate_sample_size(baseline_rate=0.10, mde=0.02, alpha=0.05, power=0.80)
        assert isinstance(n, int)
        assert n > 0

    def test_smaller_mde_needs_larger_sample(self):
        n_large_mde = calculate_sample_size(0.10, mde=0.05, alpha=0.05, power=0.80)
        n_small_mde = calculate_sample_size(0.10, mde=0.01, alpha=0.05, power=0.80)
        assert n_small_mde > n_large_mde


class TestMeasureUplift:
    def test_positive_uplift(self):
        np.random.seed(42)
        treatment = pd.DataFrame({"converted": np.random.binomial(1, 0.20, 500)})
        control = pd.DataFrame({"converted": np.random.binomial(1, 0.10, 500)})
        result = measure_uplift(treatment, control, "converted")
        assert result.absolute_uplift > 0

    def test_returns_p_value(self):
        np.random.seed(42)
        treatment = pd.DataFrame({"converted": np.random.binomial(1, 0.15, 500)})
        control = pd.DataFrame({"converted": np.random.binomial(1, 0.10, 500)})
        result = measure_uplift(treatment, control, "converted")
        assert 0 <= result.p_value <= 1


class TestMultipleTestingCorrection:
    def test_bonferroni_increases_threshold(self):
        p_values = [0.01, 0.03, 0.04, 0.06]
        corrected = correct_multiple_testing(p_values, method="bonferroni")
        # Bonferroni補正後、元のp値より大きくなる（もしくは同じ）
        for orig, corr in zip(p_values, corrected):
            assert corr >= orig
