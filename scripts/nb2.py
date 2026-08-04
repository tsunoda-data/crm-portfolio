# %% [markdown]
# # 📓 Notebook 2: 構造・時系列・外部環境分析
# 
# Notebook 1で生成・クレンジングしたデータを用いて、ファンベース分析、
# NBDモデルによる購買確率推定、コホート分析による継続率の可視化、
# および競合ポジショニングマップの作成を行います。

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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln
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

# 上位20%のインデックス
top20_idx = int(len(df_sorted) * 0.2)
top20_revenue_pct = df_sorted.loc[top20_idx, 'cumulative_spend_pct']

print(f"上位20%の顧客が全体の {top20_revenue_pct*100:.1f}% の売上を占めている")

# グラフ化
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(np.linspace(0, 100, len(df_sorted)), df_sorted['cumulative_spend_pct'] * 100, color='#a78bfa', lw=3)
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
# 【セル3】NBDモデル（負の二項分布）による購買回数モデル
# ============================================================
print("--- NBDモデルによる購買確率の推定 ---")
# 期間中の購入回数分布
x = df['order_count'].values
n_customers = len(x)

# NBDの負の対数尤度関数
def nbd_nll(params, data):
    r, alpha = params
    if r <= 0 or alpha <= 0:
        return np.inf
    # log likelihood: sum [ gammaln(r+x) - gammaln(r) - gammaln(x+1) + r*log(alpha/(alpha+1)) + x*log(1/(alpha+1)) ]
    ll = np.sum(
        gammaln(r + data) - gammaln(r) - gammaln(data + 1)
        + r * np.log(alpha / (alpha + 1))
        + data * np.log(1 / (alpha + 1))
    )
    return -ll

# 最尤推定
initial_guess = [1.0, 1.0]
result = minimize(nbd_nll, initial_guess, args=(x,), bounds=[(0.001, None), (0.001, None)], method='L-BFGS-B')

r_est, alpha_est = result.x
print(f"最適化完了: r = {r_est:.4f}, alpha = {alpha_est:.4f}")

# 予測分布の計算
max_x = int(np.percentile(x, 99))
x_vals = np.arange(0, max_x + 1)
def nbd_prob(x, r, alpha):
    from scipy.special import gamma
    # 対数で計算して指数を取る（オーバーフロー防止）
    log_p = gammaln(r + x) - gammaln(r) - gammaln(x + 1) + r * np.log(alpha / (alpha + 1)) + x * np.log(1 / (alpha + 1))
    return np.exp(log_p)

expected_probs = nbd_prob(x_vals, r_est, alpha_est)
expected_counts = expected_probs * n_customers

# 実績値の集計
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

# 顧客の初回購入月と各購入月が必要だが、本データは購入履歴詳細がないため、
# signup_dateとlast_order_dateを使用して簡易的な生存率(Retention)をヒートマップで可視化する。

df['signup_month'] = df['signup_date'].dt.to_period('M')
df['last_order_month'] = df['last_order_date'].dt.to_period('M')

# 分析期間を絞る
target_months = pd.period_range(start='2025-01', end='2025-12', freq='M')
cohort_df = df[df['signup_month'].isin(target_months)]

# コホートごとのサイズ
cohort_sizes = cohort_df.groupby('signup_month').size()

# 各コホートの生存期間（ヶ月）を計算
cohort_df['lifetime_months'] = (cohort_df['last_order_month'] - cohort_df['signup_month']).apply(lambda x: x.n if pd.notna(x) else 0)

# 生存マトリクスの作成
retention_matrix = pd.DataFrame(index=target_months, columns=range(0, 13))

for month in target_months:
    cohort_users = cohort_df[cohort_df['signup_month'] == month]
    size = len(cohort_users)
    if size == 0:
        continue
    for period in range(0, 13):
        # 指定期間以降も生存している（最後の購入が指定期間より後）顧客の割合
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

# カテゴリ別の平均購入金額と平均レビュー評価
category_stats = df.groupby('main_category').agg(
    avg_spend=('total_spend', 'mean'),
    avg_review=('avg_review_score', 'mean'),
    customer_count=('customer_id', 'count')
).dropna()

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    category_stats['avg_spend'], 
    category_stats['avg_review'],
    s=category_stats['customer_count'] * 0.5, # バブルの大きさは顧客数
    alpha=0.6,
    c=np.arange(len(category_stats)),
    cmap='Set2'
)

# ラベル付与
for i, row in category_stats.iterrows():
    ax.text(row['avg_spend'], row['avg_review'], i, fontsize=12, fontweight='bold', ha='center', va='center')

# 象限を分ける十字線
ax.axvline(category_stats['avg_spend'].mean(), color='gray', linestyle='--')
ax.axhline(category_stats['avg_review'].mean(), color='gray', linestyle='--')

ax.set_title('カテゴリ別 ポジショニングマップ（バブルサイズ＝顧客数）', fontsize=14)
ax.set_xlabel('平均累計購入金額（円）', fontsize=12)
ax.set_ylabel('平均レビュースコア', fontsize=12)
plt.tight_layout()
plt.show()
print("✅ Notebook 2 完了")
