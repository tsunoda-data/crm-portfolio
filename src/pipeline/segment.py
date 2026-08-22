"""
Segmentation module.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)

CLUSTER_FEATURES = [
    'login_days_30', 'avg_session_duration', 'favorites_count', 'brand_trend_exposure',
    'order_count', 'total_spend', 'avg_order_value', 'is_subscriber', 'review_count',
    'email_open_count', 'coupon_uses', 'abandoned_carts'
]

KEY_COLS = [
    'total_spend', 'order_count', 'avg_order_value', 'login_days_30',
    'favorites_count', 'review_count', 'is_subscriber', 'email_open_count'
]

def fit_segments(df: pd.DataFrame, n_clusters: int = 9) -> tuple[KMeans, StandardScaler, pd.DataFrame, dict]:
    """Fits K-Means and returns models and mapping rules."""
    logger.info(f"Fitting segments with {n_clusters} clusters...")
    df_ml = df[CLUSTER_FEATURES].copy().fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_ml)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_tmp = df.copy()
    df_tmp['cluster_raw'] = clusters
    
    cluster_profile = df_tmp.groupby('cluster_raw')[KEY_COLS].mean()
    mms = MinMaxScaler()
    profile_scaled = pd.DataFrame(mms.fit_transform(cluster_profile), index=cluster_profile.index, columns=cluster_profile.columns)
    
    profile_scaled['buy_score'] = profile_scaled[['total_spend', 'order_count', 'avg_order_value']].mean(axis=1)
    profile_scaled['interest_score'] = profile_scaled[['login_days_30', 'favorites_count']].mean(axis=1)
    profile_scaled['loyalty_score'] = profile_scaled[['is_subscriber', 'review_count', 'email_open_count']].mean(axis=1)
    
    thresholds = {
        'BUY_HIGH': profile_scaled['buy_score'].median(),
        'INTEREST_HIGH': profile_scaled['interest_score'].median(),
        'LOYALTY_HIGH': profile_scaled['loyalty_score'].median()
    }
    
    return kmeans, scaler, profile_scaled, thresholds

def predict_segments(df: pd.DataFrame, kmeans: KMeans, scaler: StandardScaler, profile_scaled: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Assigns segments to data using pre-fitted model."""
    logger.info("Predicting segments...")
    df_out = df.copy()
    df_ml = df_out[CLUSTER_FEATURES].copy().fillna(0)
    X_scaled = scaler.transform(df_ml)
    
    df_out['cluster_raw'] = kmeans.predict(X_scaled)
    
    b_high = thresholds['BUY_HIGH']
    i_high = thresholds['INTEREST_HIGH']
    l_high = thresholds['LOYALTY_HIGH']
    
    def assign_9seg(c):
        row = profile_scaled.loc[c]
        b, i, l = row['buy_score'], row['interest_score'], row['loyalty_score']
        if b >= b_high and l >= l_high: return 'Seg1_ロイヤル顧客'
        elif b >= b_high and i >= i_high: return 'Seg2_一般顧客'
        elif b >= b_high: return 'Seg3_離反顧客'
        elif b < b_high and i >= i_high and l >= l_high: return 'Seg4_見込み優良顧客'
        elif b < b_high and i >= i_high: return 'Seg5_見込み一般顧客'
        elif b < b_high and l >= l_high: return 'Seg6_休眠顧客'
        else: return 'Seg7_潜在認知顧客'
        
    df_out['segment_label'] = df_out['cluster_raw'].map(assign_9seg)
    return df_out

def apply_psychology_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Seg8/Seg9 overrides based on psychology bugs."""
    logger.info("Applying psychology segments...")
    df_out = df.copy()
    
    cart_threshold = df_out['abandoned_carts'].quantile(0.67) if not df_out.empty else 0
    seg8_mask = (df_out['purchase_hour_zone'] == '深夜') & (df_out['abandoned_carts'] >= cart_threshold)
    df_out.loc[seg8_mask, 'segment_label'] = 'Seg8_深夜葛藤層'
    
    if 'repurchased_after_low_review' in df_out.columns:
        seg9_mask = (df_out['repurchased_after_low_review'] == 1) & (~seg8_mask)
        df_out.loc[seg9_mask, 'segment_label'] = 'Seg9_低評価中毒層'
        
    return df_out

def get_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns summary stats per segment."""
    return df.groupby('segment_label').agg(
        n_customers=('customer_id', 'count'),
        avg_spend=('total_spend', 'mean')
    ).round(1).reset_index()
