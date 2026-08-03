# %% [markdown]
# # 📓 Notebook 1: データ生成 & クレンジング
# 
# 実務未経験からのポートフォリオ用として、10,000人規模のCRMデータを生成し、
# 意図的な汚れの混入からクレンジング、Parquet形式での保存までを一気通貫で行います。

# %%
# ============================================================
# 【セル1】ライブラリのインポートと個人情報・行動データの生成
# ============================================================
import hashlib
import warnings
import random
import numpy as np
import pandas as pd
!pip install japanize-matplotlib --quiet
import japanize_matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 乱数シード固定
np.random.seed(42)
random.seed(42)

# 定数設定
N_CUSTOMERS = 10000
START_DATE = pd.Timestamp('2023-01-01')
END_DATE = pd.Timestamp('2026-05-31')
DATE_NOW = pd.Timestamp('2026-06-01')

print("データ生成を開始します...")

# 1. 顧客属性（PII含む）の生成
customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS + 1)]
names = [f"User_{i}" for i in range(1, N_CUSTOMERS + 1)]
emails = [f"user_{i}@example.com" for i in range(1, N_CUSTOMERS + 1)]
ages = np.random.normal(35, 10, N_CUSTOMERS).astype(int)
ages = np.clip(ages, 18, 80)
genders = np.random.choice(['男性', '女性', 'その他'], N_CUSTOMERS, p=[0.4, 0.5, 0.1])
regions = np.random.choice(['東京都', '大阪府', '愛知県', '福岡県', '北海道', 'その他'], N_CUSTOMERS)

# 登録日の生成
signup_days = np.random.randint(0, (END_DATE - START_DATE).days, N_CUSTOMERS)
signup_dates = [START_DATE + pd.Timedelta(days=d) for d in signup_days]

# デバイスとチャネル
devices = np.random.choice(['iOS', 'Android', 'PC'], N_CUSTOMERS, p=[0.5, 0.3, 0.2])
channels = np.random.choice(['Organic', 'Paid Search', 'SNS', 'Referral'], N_CUSTOMERS, p=[0.4, 0.3, 0.2, 0.1])

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

# %%
# ============================================================
# 【セル2】エンゲージメント・購買データの生成（相関ルールの注入）
# ============================================================
# 基本のエンゲージメント指標
df['login_days_30'] = np.random.poisson(5, N_CUSTOMERS)
df['avg_session_duration'] = np.random.gamma(shape=2.0, scale=3.0, size=N_CUSTOMERS).round(1)
df['favorites_count'] = np.random.poisson(2, N_CUSTOMERS)

# 購入回数の生成（パレートの法則をシミュレート：一部のロイヤル層が多数購入）
# 80%は購入数少なめ、20%は購入数多め
is_fan = np.random.choice([True, False], N_CUSTOMERS, p=[0.2, 0.8])
df['is_fan'] = is_fan

order_counts = np.where(is_fan, np.random.poisson(15, N_CUSTOMERS), np.random.poisson(2, N_CUSTOMERS))
df['order_count'] = order_counts

# 平均単価と累計金額
avg_order_values = np.where(is_fan, np.random.normal(8000, 2000, N_CUSTOMERS), np.random.normal(4000, 1000, N_CUSTOMERS))
avg_order_values = np.clip(avg_order_values, 1000, 50000).astype(int)
df['total_spend'] = df['order_count'] * avg_order_values

# 定期購入（サブスク）フラグ
df['is_subscriber'] = np.where(is_fan & (df['order_count'] > 5), 1, 0)
# たまにランダムで追加
df['is_subscriber'] = np.where(np.random.random(N_CUSTOMERS) < 0.05, 1, df['is_subscriber'])

# 初回/最終購入日
def get_order_dates(row):
    if row['order_count'] == 0:
        return pd.NaT, pd.NaT
    days_since = (DATE_NOW - row['signup_date']).days
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

# その他の購買行動
hour_zones = ['朝', '昼', '夜', '深夜']
df['purchase_hour_zone'] = np.random.choice(hour_zones, N_CUSTOMERS, p=[0.15, 0.25, 0.4, 0.2])

# 深夜はカゴ落ちが多い（相関ルール）
df['abandoned_carts'] = np.where(df['purchase_hour_zone'] == '深夜', np.random.poisson(4, N_CUSTOMERS), np.random.poisson(1, N_CUSTOMERS))
df['coupon_uses'] = (df['order_count'] * np.random.uniform(0, 0.5, N_CUSTOMERS)).astype(int)

# レビューと低評価中毒層
df['review_count'] = (df['order_count'] * np.random.uniform(0, 0.3, N_CUSTOMERS)).astype(int)
df['avg_review_score'] = np.where(df['review_count'] > 0, np.random.normal(3.8, 0.8, N_CUSTOMERS), np.nan)
df['avg_review_score'] = np.clip(df['avg_review_score'], 1.0, 5.0)

# 松本理論：低評価なのにリピートする層
df['repurchased_after_low_review'] = np.where(
    (df['avg_review_score'] <= 2.5) & (df['order_count'] >= 5), 1, 0
)

# 外部環境データ
df['sns_sentiment_score'] = np.random.normal(0.5, 0.2, N_CUSTOMERS)
df['competitor_price_diff'] = np.random.normal(-500, 1500, N_CUSTOMERS)
df['brand_trend_exposure'] = np.random.randint(0, 100, N_CUSTOMERS)
df['main_category'] = np.random.choice(['アパレル', 'コスメ', '家電', '食品', '雑貨'], N_CUSTOMERS)

# メール開封数
df['email_open_count'] = np.random.poisson(10, N_CUSTOMERS) + (df['is_fan'] * 15)

df = df.drop(columns=['is_fan'])

print(f"データ生成完了: {df.shape[0]}行 × {df.shape[1]}列")

# %%
# ============================================================
# 【セル3】意図的なデータ汚れの注入（クレンジング課題）
# ============================================================
print("意図的な汚れ（欠損・外れ値・矛盾など）を注入します...")
df_dirty = df.copy()

# 1. 欠損値（Missing Values）
missing_idx = np.random.choice(df_dirty.index, 300, replace=False)
df_dirty.loc[missing_idx, 'region'] = np.nan

missing_age_idx = np.random.choice(df_dirty.index, 200, replace=False)
df_dirty.loc[missing_age_idx, 'age'] = np.nan

# 2. 外れ値（Outliers）
outlier_idx = np.random.choice(df_dirty.index, 50, replace=False)
df_dirty.loc[outlier_idx, 'total_spend'] = df_dirty.loc[outlier_idx, 'total_spend'] * 100

# 3. 表記揺れ（Inconsistent formatting）
typo_idx = np.random.choice(df_dirty.index, 100, replace=False)
df_dirty.loc[typo_idx, 'region'] = 'TOKYO'

# 4. データ型の混入（文字列としての日付）
df_dirty['signup_date'] = df_dirty['signup_date'].astype(str)

# 5. ビジネス的矛盾（購入回数0なのに累計金額がある、等）
contradict_idx = np.random.choice(df_dirty[df_dirty['order_count'] == 0].index, 50, replace=False)
df_dirty.loc[contradict_idx, 'total_spend'] = 15000

print("汚れ注入完了！")

# %%
# ============================================================
# 【セル4】データクレンジング
# ============================================================
print("データクレンジングを開始します...")
df_clean = df_dirty.copy()

# 1. 個人情報(PII)の保護：メールアドレスのハッシュ化（SHA-256）、名前カラムの削除
df_clean['customer_hash'] = df_clean['email'].apply(
    lambda x: hashlib.sha256(x.encode()).hexdigest()
)
df_clean = df_clean.drop(columns=['name', 'email'])
print("  ✅ 個人情報のハッシュ化と削除")

# 2. 型変換（日付文字列 → datetime）
df_clean['signup_date'] = pd.to_datetime(df_clean['signup_date'])
print("  ✅ 日付型の変換")

# 3. 表記揺れの修正
df_clean['region'] = df_clean['region'].replace('TOKYO', '東京都')
print("  ✅ 表記揺れの修正")

# 4. 外れ値の処理（IQR法によるキャップ処理）
Q1 = df_clean['total_spend'].quantile(0.25)
Q3 = df_clean['total_spend'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

outliers_mask = df_clean['total_spend'] > upper_bound
df_clean.loc[outliers_mask, 'total_spend'] = upper_bound
print("  ✅ 外れ値のキャップ処理 (IQR法)")

# 5. ビジネス的矛盾の修正
# 購入回数0なのに金額がある場合 → 金額を0にする
contradictions = (df_clean['order_count'] == 0) & (df_clean['total_spend'] > 0)
df_clean.loc[contradictions, 'total_spend'] = 0
print("  ✅ ビジネス矛盾の修正")

# 6. 欠損値の処理
df_clean['region'] = df_clean['region'].fillna('不明')
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
print("  ✅ 欠損値の補完")

# 7. 派生変数の再計算
df_clean['avg_order_value'] = np.where(
    df_clean['order_count'] > 0, 
    df_clean['total_spend'] / df_clean['order_count'], 
    0
)
print("  ✅ 派生変数の再計算")

# %%
# ============================================================
# 【セル5】品質確認とParquet形式での保存
# ============================================================
import os

print("--- 最終データの品質確認 ---")
assert df_clean['total_spend'].max() <= upper_bound, "外れ値が残っています"
assert df_clean['age'].isnull().sum() == 0, "年齢に欠損値が残っています"
assert ((df_clean['order_count'] == 0) & (df_clean['total_spend'] > 0)).sum() == 0, "矛盾が残っています"
assert 'email' not in df_clean.columns, "平文のメールアドレスが残っています"

print("  ✅ アサーションテスト全パス！")

# 保存ディレクトリの作成
OUTPUT_DIR = '/Users/user/projects/crm-portfolio/notebooks/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)
PARQUET_PATH = os.path.join(OUTPUT_DIR, 'notebook1_cleaned_data.parquet')

# Parquet形式で保存（型情報を維持してファイルサイズも軽量）
df_clean.to_parquet(PARQUET_PATH, engine='pyarrow', index=False)

print(f"✅ データを保存しました: {PARQUET_PATH}")
print(f"   データ形状: {df_clean.shape[0]}行 × {df_clean.shape[1]}列")
