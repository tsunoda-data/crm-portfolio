# %% [markdown]
# # 📓 Notebook 3: 顧客セグメンテーション（9セグマップ）
# 
# 行動データを用いたK-Meansクラスタリングにより、顧客を「9セグマップ」へマッピングします。
# 松本健太郎理論に基づく「心理バグセグメント（葛藤層・中毒層）」も抽出します。

# %%
# ============================================================
# 【セル1】ライブラリインポート・データ読み込み・前処理
# ============================================================
import numpy as np
import pandas as pd
import sys
import subprocess

try:
    import japanize_matplotlib
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "japanize-matplotlib", "--quiet"])
    import japanize_matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# plt.rcParams['font.family'] = 'IPAexGothic'
plt.rcParams['figure.facecolor'] = '#0f1117'
plt.rcParams['axes.facecolor'] = '#1a1d2e'
plt.rcParams['axes.edgecolor'] = '#3a3d4f'
plt.rcParams['text.color'] = '#e8eaf0'
plt.rcParams['axes.labelcolor'] = '#e8eaf0'
plt.rcParams['xtick.color'] = '#a0a4b8'
plt.rcParams['ytick.color'] = '#a0a4b8'
plt.rcParams['grid.color'] = '#2a2d3e'
plt.rcParams['grid.alpha'] = 0.5

PARQUET_PATH = '/Users/user/projects/crm-portfolio/notebooks/data/notebook1_cleaned_data.parquet'
df = pd.read_parquet(PARQUET_PATH, engine='pyarrow')

CLUSTER_FEATURES = [
    'login_days_30', 'avg_session_duration', 'favorites_count', 'brand_trend_exposure',
    'order_count', 'total_spend', 'avg_order_value', 'is_subscriber', 'review_count',
    'email_open_count', 'coupon_uses', 'abandoned_carts'
]

df_ml = df[CLUSTER_FEATURES].copy()
if 'avg_review_score' in df.columns:
    df_ml['avg_review_score'] = df['avg_review_score'].fillna(0)

df_ml = df_ml.fillna(0)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_ml)
print(f"✅ データ準備と標準化完了: {X_scaled.shape}")

# %%
# ============================================================
# 【セル2】最適クラスタ数の決定（エルボー法 & シルエット分析）
# ============================================================
print("--- K-Means最適クラスタ数の探索 ---")
K_RANGE = range(2, 13)
inertias = []
sil_scores = []

for k in K_RANGE:
    kmeans_k = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = kmeans_k.fit_predict(X_scaled)
    inertias.append(kmeans_k.inertia_)
    sil = silhouette_score(X_scaled, labels_k, sample_size=3000, random_state=42)
    sil_scores.append(sil)

elbow_k = 4 # 簡易的に4とする（ロジックで出す場合は2階差分）
best_sil_k = list(K_RANGE)[np.argmax(sil_scores)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(list(K_RANGE), inertias, 'o-', color='#38bdf8')
ax1.axvline(9, color='#a78bfa', linestyle='--', label='採用: K=9')
ax1.set_title('エルボー法')
ax1.legend()

ax2.bar(list(K_RANGE), sil_scores, color='#38bdf8')
ax2.set_title('シルエット分析')
plt.show()

print(f"9セグマップに合わせて K=9 を採用します。")

# %%
# ============================================================
# 【セル3】K-Means (K=9) 実行と 9セグへのマッピング
# ============================================================
print("--- K=9 クラスタリングと9セグマップへのマッピング ---")
kmeans_final = KMeans(n_clusters=9, n_init=20, random_state=42)
df['cluster_raw'] = kmeans_final.fit_predict(X_scaled)

KEY_COLS = [
    'total_spend', 'order_count', 'avg_order_value', 'login_days_30',
    'favorites_count', 'review_count', 'is_subscriber', 'email_open_count'
]
cluster_profile = df.groupby('cluster_raw')[KEY_COLS].mean()
mms = MinMaxScaler()
profile_scaled = pd.DataFrame(mms.fit_transform(cluster_profile), index=cluster_profile.index, columns=cluster_profile.columns)

profile_scaled['buy_score'] = profile_scaled[['total_spend', 'order_count', 'avg_order_value']].mean(axis=1)
profile_scaled['interest_score'] = profile_scaled[['login_days_30', 'favorites_count']].mean(axis=1)
profile_scaled['loyalty_score'] = profile_scaled[['is_subscriber', 'review_count', 'email_open_count']].mean(axis=1)

BUY_HIGH = profile_scaled['buy_score'].median()
INTEREST_HIGH = profile_scaled['interest_score'].median()
LOYALTY_HIGH = profile_scaled['loyalty_score'].median()

def assign_9seg(row):
    b, i, l = row['buy_score'], row['interest_score'], row['loyalty_score']
    if b >= BUY_HIGH and l >= LOYALTY_HIGH: return 'Seg1_ロイヤル顧客'
    elif b >= BUY_HIGH and i >= INTEREST_HIGH: return 'Seg2_一般顧客'
    elif b >= BUY_HIGH: return 'Seg3_離反顧客'
    elif b < BUY_HIGH and i >= INTEREST_HIGH and l >= LOYALTY_HIGH: return 'Seg4_見込み優良顧客'
    elif b < BUY_HIGH and i >= INTEREST_HIGH: return 'Seg5_見込み一般顧客'
    elif b < BUY_HIGH and l >= LOYALTY_HIGH: return 'Seg6_休眠顧客'
    else: return 'Seg7_潜在認知顧客'

df['segment_label'] = df['cluster_raw'].map(lambda c: assign_9seg(profile_scaled.loc[c]))
print("基本セグメントの割り当て完了。")

# %%
# ============================================================
# 【セル4】心理バグセグメント（松本理論）の抽出と上書き
# ============================================================
print("--- 松本理論：心理バグセグメントの抽出 ---")
cart_threshold = df['abandoned_carts'].quantile(0.67)
seg8_mask = (df['purchase_hour_zone'] == '深夜') & (df['abandoned_carts'] >= cart_threshold)
df.loc[seg8_mask, 'segment_label'] = 'Seg8_深夜葛藤層'

seg9_mask = (df['repurchased_after_low_review'] == 1) & (~seg8_mask)
df.loc[seg9_mask, 'segment_label'] = 'Seg9_低評価中毒層'

print(df['segment_label'].value_counts())

# %%
# ============================================================
# 【セル5】可視化（PCA散布図 & レーダーチャート）
# ============================================================
print("--- 9セグマップのPCA可視化 ---")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['pca_x'], df['pca_y'] = X_pca[:, 0], X_pca[:, 1]

fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=df, x='pca_x', y='pca_y', hue='segment_label', palette='tab10', alpha=0.6, ax=ax)
ax.set_title('9セグマップPCA散布図', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 【セル6】施策テーブル出力とParquet保存
# ============================================================
import os
print("--- セグメント別施策テーブルの出力と保存 ---")
seg_summary = df.groupby('segment_label').agg(
    人数=('customer_id', 'count'),
    平均購入金額=('total_spend', 'mean')
).round(1)
display(seg_summary)

OUTPUT_PATH = '/Users/user/projects/crm-portfolio/notebooks/data/notebook3_segmented_data.parquet'
df.to_parquet(OUTPUT_PATH, index=False, engine='pyarrow')
print(f"✅ Notebook 3 完了。データを保存しました: {OUTPUT_PATH}")
