"""
Data ingestion and cleansing module.
"""
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def generate_synthetic(cfg: dict) -> pd.DataFrame:
    """Generates synthetic customer data."""
    logger.info("Generating synthetic data...")
    syn_cfg = cfg.get("synthetic", {})
    n_customers = syn_cfg.get("n_customers", 10000)
    start_date = pd.Timestamp(syn_cfg.get("start_date", "2023-01-01"))
    end_date = pd.Timestamp(syn_cfg.get("end_date", "2026-05-31"))
    seed = syn_cfg.get("random_seed", 42)
    date_now = pd.Timestamp(cfg.get("pipeline", {}).get("run_date", "2026-06-01"))

    np.random.seed(seed)
    
    customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, n_customers + 1)]
    names = [f"User_{i}" for i in range(1, n_customers + 1)]
    emails = [f"user_{i}@example.com" for i in range(1, n_customers + 1)]
    ages = np.random.normal(35, 10, n_customers).astype(int)
    ages = np.clip(ages, 18, 80)
    genders = np.random.choice(['男性', '女性', 'その他'], n_customers, p=[0.4, 0.5, 0.1])
    regions = np.random.choice(['東京都', '大阪府', '愛知県', '福岡県', '北海道', 'その他'], n_customers)

    signup_days = np.random.randint(0, (end_date - start_date).days, n_customers)
    signup_dates = [start_date + pd.Timedelta(days=d) for d in signup_days]

    devices = np.random.choice(['iOS', 'Android', 'PC'], n_customers, p=[0.5, 0.3, 0.2])
    channels = np.random.choice(['Organic', 'Paid Search', 'SNS', 'Referral'], n_customers, p=[0.4, 0.3, 0.2, 0.1])

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'name': names,
        'email': emails,
        'age': ages,
        'gender': genders,
        'region': regions,
        'signup_date': signup_dates,
        'main_device': devices,
        'traffic_channel': channels
    })

    df['login_days_30'] = np.random.poisson(5, n_customers)
    df['avg_session_duration'] = np.random.gamma(shape=2.0, scale=3.0, size=n_customers).round(1)
    df['favorites_count'] = np.random.poisson(2, n_customers)

    is_fan = np.random.choice([True, False], n_customers, p=[0.2, 0.8])
    order_counts = np.where(is_fan, np.random.poisson(15, n_customers), np.random.poisson(2, n_customers))
    df['order_count'] = order_counts

    avg_order_values = np.where(is_fan, np.random.normal(8000, 2000, n_customers), np.random.normal(4000, 1000, n_customers))
    avg_order_values = np.clip(avg_order_values, 1000, 50000).astype(int)
    df['total_spend'] = df['order_count'] * avg_order_values

    df['is_subscriber'] = np.where(is_fan & (df['order_count'] > 5), 1, 0)
    df['is_subscriber'] = np.where(np.random.random(n_customers) < 0.05, 1, df['is_subscriber'])

    def get_order_dates(row):
        if row['order_count'] == 0:
            return pd.NaT, pd.NaT
        days_since = (date_now - row['signup_date']).days
        if days_since < 1:
            days_since = 1
        first_days = np.random.randint(0, min(30, days_since))
        first = row['signup_date'] + pd.Timedelta(days=first_days)
        last_days = np.random.randint(first_days, days_since)
        last = row['signup_date'] + pd.Timedelta(days=last_days)
        return first, last

    dates = df.apply(get_order_dates, axis=1)
    df['first_order_date'] = [d[0] for d in dates]
    df['last_order_date'] = [d[1] for d in dates]

    hour_zones = ['朝', '昼', '夜', '深夜']
    df['purchase_hour_zone'] = np.random.choice(hour_zones, n_customers, p=[0.15, 0.25, 0.4, 0.2])

    df['abandoned_carts'] = np.where(df['purchase_hour_zone'] == '深夜', np.random.poisson(4, n_customers), np.random.poisson(1, n_customers))
    df['coupon_uses'] = (df['order_count'] * np.random.uniform(0, 0.5, n_customers)).astype(int)

    df['review_count'] = (df['order_count'] * np.random.uniform(0, 0.3, n_customers)).astype(int)
    df['avg_review_score'] = np.where(df['review_count'] > 0, np.random.normal(3.8, 0.8, n_customers), np.nan)
    df['avg_review_score'] = np.clip(df['avg_review_score'], 1.0, 5.0)

    df['repurchased_after_low_review'] = np.where((df['avg_review_score'] <= 2.5) & (df['order_count'] >= 5), 1, 0)
    
    df['sns_sentiment_score'] = np.random.normal(0.5, 0.2, n_customers)
    df['competitor_price_diff'] = np.random.normal(-500, 1500, n_customers)
    df['brand_trend_exposure'] = np.random.randint(0, 100, n_customers)
    df['main_category'] = np.random.choice(['アパレル', 'コスメ', '家電', '食品', '雑貨'], n_customers)
    
    df['email_open_count'] = np.random.poisson(10, n_customers) + (is_fan * 15)

    missing_idx = np.random.choice(df.index, 300, replace=False)
    df.loc[missing_idx, 'region'] = np.nan
    missing_age_idx = np.random.choice(df.index, 200, replace=False)
    df.loc[missing_age_idx, 'age'] = np.nan
    outlier_idx = np.random.choice(df.index, 50, replace=False)
    df.loc[outlier_idx, 'total_spend'] = df.loc[outlier_idx, 'total_spend'] * 100
    typo_idx = np.random.choice(df.index, 100, replace=False)
    df.loc[typo_idx, 'region'] = 'TOKYO'
    df['signup_date'] = df['signup_date'].astype(str)
    contradict_idx = np.random.choice(df[df['order_count'] == 0].index, 50, replace=False)
    df.loc[contradict_idx, 'total_spend'] = 15000

    logger.info(f"Synthetic data generated: {df.shape}")
    return df

def load_from_parquet(path: str | Path) -> pd.DataFrame:
    """Loads existing data from Parquet."""
    logger.info(f"Loading data from {path}")
    return pd.read_parquet(path, engine='pyarrow')

def cleanse(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Cleanses raw dataframe."""
    logger.info("Cleansing data...")
    df_clean = df.copy()

    if 'email' in df_clean.columns:
        df_clean['customer_hash'] = df_clean['email'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
        df_clean = df_clean.drop(columns=['name', 'email'], errors='ignore')

    df_clean['signup_date'] = pd.to_datetime(df_clean['signup_date'])
    df_clean['region'] = df_clean['region'].replace('TOKYO', '東京都')

    Q1 = df_clean['total_spend'].quantile(0.25)
    Q3 = df_clean['total_spend'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    # pandas 2.x では型の一致が厳格。total_spend の dtype に合わせてキャスト
    upper_bound_casted = int(upper_bound) if df_clean['total_spend'].dtype == 'int64' else upper_bound
    outliers_mask = df_clean['total_spend'] > upper_bound
    df_clean.loc[outliers_mask, 'total_spend'] = upper_bound_casted

    contradictions = (df_clean['order_count'] == 0) & (df_clean['total_spend'] > 0)
    df_clean.loc[contradictions, 'total_spend'] = 0

    df_clean['region'] = df_clean['region'].fillna('不明')
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())

    df_clean['avg_order_value'] = np.where(
        df_clean['order_count'] > 0,
        df_clean['total_spend'] / df_clean['order_count'],
        0
    )

    logger.info(f"Cleansing completed: {df_clean.shape}")
    return df_clean

def save_checkpoint(df: pd.DataFrame, name: str, cfg: dict) -> None:
    """Saves DataFrame to data_dir."""
    data_dir = Path(cfg.get("pipeline", {}).get("data_dir", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    run_date = cfg.get("pipeline", {}).get("run_date", "auto")
    
    path = data_dir / f"{name}_{run_date}.parquet"
    logger.info(f"Saving checkpoint to {path}")
    df.to_parquet(path, engine='pyarrow', index=False)
