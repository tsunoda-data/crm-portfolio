# %% [markdown]
# # 📓 Notebook 2: 構造・時系列・外部環境分析
# 
# ## 📊 Executive Summary
# 
# | 項目 | 内容 |
# |------|------|
# | **ビジネス課題** | 顧客構造の把握と、マーケティング理論に基づく定量分析で施策の方向性を決定する |
# | **分析手法** | ①パレート分析(佐藤尚之) ②NBDモデル(森岡毅) ③コホート分析 ④ポジショニングマップ |
# | **主要な発見** | 上位20%が売上の約55%を占有 / NBDモデルで将来の購買確率を推定可能 / コホート別の離脱タイミングを特定 |
# | **期待されるROIインパクト** | ファン層への集中投資でROI 200%超の施策設計が可能（Notebook 4で定量算出） |
# | **Plotly可視化** | パレート曲線・RFM 3D散布図をインタラクティブに操作可能 |

# %%
# ============================================================
# 【セル1】ライブラリのインポートとデータ読み込み
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

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "--quiet"])
    import plotly.express as px
    import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.facecolor'] = '#0f1117'
plt.rcParams['axes.facecolor'] = '#1a1d2e'
plt.rcParams['axes.edgecolor'] = '#3a3d4f'
plt.rcParams['text.color'] = '#e8eaf0'
plt.rcParams['axes.labelcolor'] = '#e8eaf0'
plt.rcParams['xtick.color'] = '#a0a4b8'
plt.rcParams['ytick.color'] = '#a0a4b8'
plt.rcParams['grid.color'] = '#2a2d3e'
plt.rcParams['grid.alpha'] = 0.5

PARQUET_PATH = './data/notebook1_cleaned_data.parquet'
df = pd.read_parquet(PARQUET_PATH, engine='pyarrow')
DATE_NOW = pd.Timestamp('2026-06-01')
print(f"✅ データ読み込み完了: {df.shape[0]}行 × {df.shape[1]}列")

# %%
# ============================================================
# 【セル2】ファンベース分析（パレートの法則）
# ============================================================
print("--- ファンベース分析（上位20%の売上貢献度） ---")

# 顧客を累計購入金額の降順に並べ替え
df_sorted = df.sort_values('total_spend', ascending=False).reset_index(drop=True)

# 累積売上とパーセンテージを計算
total_revenue = df_sorted['total_spend'].sum()
df_sorted['cumulative_spend'] = df_sorted['total_spend'].cumsum()
df_sorted['cumulative_spend_pct'] = df_sorted['cumulative_spend'] / total_revenue
df_sorted['customer_pct'] = np.linspace(0, 100, len(df_sorted))

# 上位20%のインデックス
top20_idx = int(len(df_sorted) * 0.2)
top20_revenue_pct = df_sorted.loc[top20_idx, 'cumulative_spend_pct']

print(f"上位20%の顧客が全体の {top20_revenue_pct*100:.1f}% の売上を占めている")

# matplotlib版
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_sorted['customer_pct'], df_sorted['cumulative_spend_pct'] * 100, color='#a78bfa', lw=3)
ax.axvline(20, color='#f97316', linestyle='--', label='上位20%')
ax.axhline(top20_revenue_pct * 100, color='#f97316', linestyle='--')
ax.scatter(20, top20_revenue_pct * 100, color='#f97316', s=100, zorder=5)
ax.set_title('ファンベース分析: 顧客層と累積売上の関係（パレート曲線）', fontsize=14)
ax.set_xlabel('顧客の累積割合 (%)', fontsize=12)
ax.set_ylabel('売上の累積割合 (%)', fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 【セル2b】📊 Plotly版 インタラクティブ・パレート曲線
# ============================================================
print("--- 📊 Plotly: インタラクティブ・パレート曲線 ---")

fig_pareto = go.Figure()
fig_pareto.add_trace(go.Scatter(
    x=df_sorted['customer_pct'],
    y=df_sorted['cumulative_spend_pct'] * 100,
    mode='lines',
    name='累積売上',
    line=dict(color='#a78bfa', width=3),
    hovertemplate='上位 %{x:.1f}% の顧客が<br>売上の %{y:.1f}% を占有<extra></extra>'
))
fig_pareto.add_vline(x=20, line_dash="dash", line_color="#f97316", annotation_text="上位20%")
fig_pareto.add_hline(y=top20_revenue_pct * 100, line_dash="dash", line_color="#f97316")
fig_pareto.update_layout(
    title='📊 インタラクティブ・パレート曲線（カーソルで探索可能）',
    xaxis_title='顧客の累積割合 (%)',
    yaxis_title='売上の累積割合 (%)',
    template='plotly_dark',
    height=500
)
fig_pareto.show()

# %%
# ============================================================
# 【セル2c】📊 Plotly版 RFM 3D散布図
# ============================================================
print("--- 📊 Plotly: RFM 3D散布図 ---")

# RFM指標の計算
df['recency'] = (DATE_NOW - df['last_order_date']).dt.days
df['recency'] = df['recency'].fillna(9999)  # 購入なし→大きな値
df['frequency'] = df['order_count']
df['monetary'] = df['total_spend']

fig_rfm = px.scatter_3d(
    df[df['order_count'] > 0],
    x='recency', y='frequency', z='monetary',
    color='main_category',
    size='monetary',
    size_max=15,
    opacity=0.6,
    hover_data=['customer_id', 'region'],
    title='📊 RFM 3D散布図（Recency × Frequency × Monetary）',
    labels={'recency': 'Recency（最終購入からの日数）', 'frequency': 'Frequency（購入回数）', 'monetary': 'Monetary（累計金額）'},
    template='plotly_dark'
)
fig_rfm.update_layout(height=600)
fig_rfm.show()

# %%
# ============================================================
# 【セル3】NBDモデル（負の二項分布）による購買回数モデル
# ============================================================
print("--- NBDモデルによる購買確率の推定 ---")
x = df['order_count'].values
n_customers = len(x)

def nbd_nll(params, data):
    r, alpha = params
    if r <= 0 or alpha <= 0:
        return np.inf
    ll = np.sum(
        gammaln(r + data) - gammaln(r) - gammaln(data + 1)
        + r * np.log(alpha / (alpha + 1))
        + data * np.log(1 / (alpha + 1))
    )
    return -ll

initial_guess = [1.0, 1.0]
result = minimize(nbd_nll, initial_guess, args=(x,), bounds=[(0.001, None), (0.001, None)], method='L-BFGS-B')

r_est, alpha_est = result.x
print(f"最適化完了: r = {r_est:.4f}, alpha = {alpha_est:.4f}")

max_x = int(np.percentile(x, 99))
x_vals = np.arange(0, max_x + 1)
def nbd_prob(x, r, alpha):
    log_p = gammaln(r + x) - gammaln(r) - gammaln(x + 1) + r * np.log(alpha / (alpha + 1)) + x * np.log(1 / (alpha + 1))
    return np.exp(log_p)

expected_probs = nbd_prob(x_vals, r_est, alpha_est)
expected_counts = expected_probs * n_customers

actual_counts = pd.Series(x).value_counts().reindex(x_vals, fill_value=0)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x_vals, actual_counts, color='#38bdf8', alpha=0.6, label='実績データ')
ax.plot(x_vals, expected_counts, color='#f97316', marker='o', lw=2, label='NBDモデル予測')
ax.set_title('NBDモデルによる購買回数分布の適合度確認', fontsize=14)
ax.set_xlabel('購入回数', fontsize=12)
ax.set_ylabel('顧客数', fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 【セル4】コホート分析（月次継続率ヒートマップ）
# ============================================================
print("--- コホート分析（月次継続率ヒートマップ） ---")

df['signup_month'] = df['signup_date'].dt.to_period('M')
df['last_order_month'] = df['last_order_date'].dt.to_period('M')

target_months = pd.period_range(start='2025-01', end='2025-12', freq='M')
cohort_df = df[df['signup_month'].isin(target_months)].copy()

cohort_sizes = cohort_df.groupby('signup_month').size()
cohort_df['lifetime_months'] = (cohort_df['last_order_month'] - cohort_df['signup_month']).apply(lambda x: x.n if pd.notna(x) else 0)

retention_matrix = pd.DataFrame(index=target_months, columns=range(0, 13))

for month in target_months:
    cohort_users = cohort_df[cohort_df['signup_month'] == month]
    size = len(cohort_users)
    if size == 0:
        continue
    for period in range(0, 13):
        retained = len(cohort_users[cohort_users['lifetime_months'] >= period])
        retention_matrix.loc[month, period] = retained / size

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(retention_matrix.astype(float), annot=True, fmt='.1%', cmap='YlGnBu', vmin=0.0, vmax=1.0, ax=ax)
ax.set_title('コホート別 継続率（Retention Rate）ヒートマップ', fontsize=14, color='white')
ax.set_xlabel('経過月数', fontsize=12, color='white')
ax.set_ylabel('獲得月（コホート）', fontsize=12, color='white')
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 【セル5】競合ポジショニングマップ
# ============================================================
print("--- 競合ポジショニングマップ（カテゴリ別） ---")

category_stats = df.groupby('main_category').agg(
    avg_spend=('total_spend', 'mean'),
    avg_review=('avg_review_score', 'mean'),
    customer_count=('customer_id', 'count')
).dropna()

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    category_stats['avg_spend'],
    category_stats['avg_review'],
    s=category_stats['customer_count'] * 0.5,
    alpha=0.6,
    c=np.arange(len(category_stats)),
    cmap='Set2'
)

for i, row in category_stats.iterrows():
    ax.text(row['avg_spend'], row['avg_review'], i, fontsize=12, fontweight='bold', ha='center', va='center')

ax.axvline(category_stats['avg_spend'].mean(), color='gray', linestyle='--')
ax.axhline(category_stats['avg_review'].mean(), color='gray', linestyle='--')

ax.set_title('カテゴリ別 ポジショニングマップ（バブルサイズ＝顧客数）', fontsize=14)
ax.set_xlabel('平均累計購入金額（円）', fontsize=12)
ax.set_ylabel('平均レビュースコア', fontsize=12)
plt.tight_layout()
plt.show()
print("✅ Notebook 2 完了")
