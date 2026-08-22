"""
Feature engineering module.
"""
import logging
from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

CHURN_FEATURE_COLUMNS = [
    'order_count', 'total_spend', 'avg_order_value', 'coupon_uses', 'coupon_rate', 'purchase_frequency',
    'login_days_30', 'avg_session_duration', 'favorites_count', 'abandoned_carts', 'review_count',
    'email_open_count', 'is_subscriber', 'repurchased_after_low_review', 'days_since_signup',
    'purchase_span_days', 'email_open_per_month', 'main_category_enc', 'main_device_enc',
    'traffic_channel_enc', 'purchase_hour_zone_enc', 'segment_label_enc', 'sns_sentiment_score',
    'competitor_price_diff', 'brand_trend_exposure', 'avg_review_score_filled'
]

LTV_FEATURE_COLUMNS = CHURN_FEATURE_COLUMNS.copy()
LTV_FEATURE_COLUMNS.remove('total_spend')
LTV_FEATURE_COLUMNS.append('age')

_fitted_encoders: dict[str, LabelEncoder] = {}

def build_rfm_features(df: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """Builds RFM features (Recency, Frequency, Monetary)."""
    logger.info("Building RFM features...")
    df_out = df.copy()
    date_now = pd.Timestamp(run_date)
    
    df_out['recency'] = (date_now - df_out['last_order_date']).dt.days.fillna(9999)
    df_out['frequency'] = df_out['order_count']
    df_out['monetary'] = df_out['total_spend']
    
    return df_out

def build_churn_features(df: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """Creates derived features for churn prediction."""
    logger.info("Building churn features...")
    df_out = df.copy()
    date_now = pd.Timestamp(run_date)
    
    df_out['days_since_last_order'] = (date_now - df_out['last_order_date']).dt.days
    df_out['churned'] = (df_out['days_since_last_order'] >= 90).astype(int)
    
    categorical_cols = ['main_category', 'main_device', 'traffic_channel', 'purchase_hour_zone', 'segment_label']
    for col in categorical_cols:
        if col not in df_out.columns:
            continue
        col_str = df_out[col].astype(str)
        if col not in _fitted_encoders:
            le = LabelEncoder()
            df_out[f'{col}_enc'] = le.fit_transform(col_str)
            _fitted_encoders[col] = le
        else:
            le = _fitted_encoders[col]
            classes = list(le.classes_)
            df_out[f'{col}_enc'] = df_out[col].map(lambda s: le.transform([s])[0] if s in classes else -1)
            
    df_out['days_since_signup'] = (date_now - df_out['signup_date']).dt.days
    df_out['purchase_span_days'] = (df_out['last_order_date'] - df_out['first_order_date']).dt.days.fillna(0)
    df_out['purchase_frequency'] = np.where(df_out['purchase_span_days'] > 0, df_out['order_count'] / df_out['purchase_span_days'], 0)
    df_out['coupon_rate'] = np.where(df_out['order_count'] > 0, df_out['coupon_uses'] / df_out['order_count'], 0)
    df_out['email_open_per_month'] = df_out['email_open_count'] / (df_out['days_since_signup'] / 30 + 1)
    df_out['avg_review_score_filled'] = df_out['avg_review_score'].fillna(df_out['avg_review_score'].median())

    return df_out
