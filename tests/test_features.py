"""
特徴量エンジニアリングのテスト
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.pipeline.features import build_churn_features, build_rfm_features, CHURN_FEATURE_COLUMNS


@pytest.fixture
def sample_df():
    """テスト用の最小DataFrameを生成。"""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "age": np.random.randint(18, 70, n),
        "gender": np.random.choice(["男性", "女性"], n),
        "region": np.random.choice(["東京都", "大阪府"], n),
        "signup_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "main_device": np.random.choice(["iOS", "Android"], n),
        "traffic_channel": np.random.choice(["Organic", "Paid Search"], n),
        "login_days_30": np.random.poisson(5, n),
        "avg_session_duration": np.random.gamma(2, 3, n).round(1),
        "favorites_count": np.random.poisson(2, n),
        "order_count": np.random.poisson(5, n),
        "total_spend": np.random.randint(0, 100000, n),
        "avg_order_value": np.random.randint(1000, 10000, n),
        "is_subscriber": np.random.choice([0, 1], n),
        "first_order_date": pd.date_range("2024-02-01", periods=n, freq="D"),
        "last_order_date": pd.date_range("2025-06-01", periods=n, freq="D"),
        "purchase_hour_zone": np.random.choice(["朝", "昼", "夜", "深夜"], n),
        "abandoned_carts": np.random.poisson(2, n),
        "coupon_uses": np.random.randint(0, 5, n),
        "review_count": np.random.randint(0, 10, n),
        "avg_review_score": np.random.uniform(1, 5, n),
        "repurchased_after_low_review": np.random.choice([0, 1], n),
        "sns_sentiment_score": np.random.normal(0.5, 0.2, n),
        "competitor_price_diff": np.random.normal(0, 1000, n),
        "brand_trend_exposure": np.random.randint(0, 100, n),
        "main_category": np.random.choice(["アパレル", "コスメ"], n),
        "email_open_count": np.random.poisson(10, n),
        "customer_hash": [f"hash_{i}" for i in range(n)],
        "segment_label": np.random.choice(["Seg1_ロイヤル顧客", "Seg3_離反顧客"], n),
    })
    return df


class TestBuildChurnFeatures:
    def test_adds_churned_column(self, sample_df):
        result = build_churn_features(sample_df, "2026-06-01")
        assert "churned" in result.columns

    def test_adds_derived_features(self, sample_df):
        result = build_churn_features(sample_df, "2026-06-01")
        expected_cols = ["days_since_last_order", "purchase_frequency", "coupon_rate", "email_open_per_month"]
        for col in expected_cols:
            assert col in result.columns, f"{col} が欠落"

    def test_churned_is_binary(self, sample_df):
        result = build_churn_features(sample_df, "2026-06-01")
        purchasers = result[result["order_count"] > 0]
        assert set(purchasers["churned"].unique()).issubset({0, 1})

    def test_no_nan_in_filled_review(self, sample_df):
        result = build_churn_features(sample_df, "2026-06-01")
        assert result["avg_review_score_filled"].isna().sum() == 0


class TestBuildRFMFeatures:
    def test_adds_rfm_columns(self, sample_df):
        result = build_rfm_features(sample_df, "2026-06-01")
        assert "recency" in result.columns
        assert "frequency" in result.columns
        assert "monetary" in result.columns

    def test_recency_is_positive(self, sample_df):
        result = build_rfm_features(sample_df, "2026-06-01")
        assert (result["recency"] >= 0).all()


class TestFeatureColumns:
    def test_churn_features_defined(self):
        assert len(CHURN_FEATURE_COLUMNS) > 0
        assert "order_count" in CHURN_FEATURE_COLUMNS
        assert "total_spend" in CHURN_FEATURE_COLUMNS
