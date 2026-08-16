# %% [markdown]
# # 📓 Notebook 4: ML予測 & ROI収益シミュレーション & AI自動化
# 
# ## 📊 Executive Summary
# 
# | 項目 | 内容 |
# |------|------|
# | **ビジネス課題** | 離脱リスクの高い顧客を事前に特定し、最適な施策でLTV最大化・離脱防止を実現する |
# | **分析手法** | LightGBM(離脱予測 AUC 0.85+) / LightGBM(LTV回帰 R²) / クーポンROI最適化 / 離脱防止収益シミュレーション |
# | **主要な発見** | 高リスク離脱層の特定 → セグメント別の最適割引率算出 → 施策別ROI比較で投資判断可能 |
# | **期待されるROIインパクト** | 離脱防止施策で年間 ¥1,500万〜¥4,000万の売上回収（3シナリオで算出） |
# | **Plotly可視化** | ROI感度分析ヒートマップ・施策インパクトウォーターフォール図 |

# %%
# ============================================================
# 【セル1】ライブラリインポート・データ準備
# ============================================================
import numpy as np
import pandas as pd
import sys
import subprocess
import os

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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
import lightgbm as lgb
import json, time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 📌 ビジネスパラメータ（調整可能な変数）
# ============================================================
MAIL_COST_PER_SEND = 3           # メール配信コスト: ¥3/通
COUPON_USAGE_RATE = 0.40         # クーポン利用率: 40%
RETENTION_RATE_EMAIL = 0.08      # 離脱防止成功率: メール 8%
RETENTION_RATE_COUPON = 0.15     # 離脱防止成功率: クーポン 15%
RETENTION_RATE_PERSONALIZED = 0.22  # 離脱防止成功率: パーソナライズメール 22%
CAC = 5000                       # 顧客獲得コスト(CAC): ¥5,000/人
CHURN_THRESHOLD = 0.6            # 離脱リスク閾値（確率≥0.6で高リスク）
DISCOUNT_RATES = [0.05, 0.10, 0.15, 0.20, 0.30]

print("📌 ビジネスパラメータ:")
print(f"   メール配信コスト: ¥{MAIL_COST_PER_SEND}/通")
print(f"   クーポン利用率: {COUPON_USAGE_RATE*100:.0f}%")
print(f"   離脱防止成功率: メール {RETENTION_RATE_EMAIL*100:.0f}% / クーポン {RETENTION_RATE_COUPON*100:.0f}% / パーソナライズ {RETENTION_RATE_PERSONALIZED*100:.0f}%")
print(f"   顧客獲得コスト(CAC): ¥{CAC:,}")

# ============================================================
# データ読み込み
# ============================================================
PARQUET_PATH = './data/notebook3_segmented_data.parquet'
df = pd.read_parquet(PARQUET_PATH, engine='pyarrow')
DATE_NOW = pd.Timestamp('2026-06-01')

# 離脱フラグの作成（購入者のみ対象、90日未購入で離脱）
df_churn = df[df['order_count'] > 0].copy()
df_churn['days_since_last_order'] = (DATE_NOW - df_churn['last_order_date']).dt.days
df_churn['churned'] = (df_churn['days_since_last_order'] >= 90).astype(int)

# カテゴリのエンコードと派生変数
for col in ['main_category', 'main_device', 'traffic_channel', 'purchase_hour_zone', 'segment_label']:
    df_churn[f'{col}_enc'] = LabelEncoder().fit_transform(df_churn[col].astype(str))

df_churn['days_since_signup'] = (DATE_NOW - df_churn['signup_date']).dt.days
df_churn['purchase_span_days'] = (df_churn['last_order_date'] - df_churn['first_order_date']).dt.days.fillna(0)
df_churn['purchase_frequency'] = np.where(df_churn['purchase_span_days'] > 0, df_churn['order_count'] / df_churn['purchase_span_days'], 0)
df_churn['coupon_rate'] = np.where(df_churn['order_count'] > 0, df_churn['coupon_uses'] / df_churn['order_count'], 0)
df_churn['email_open_per_month'] = df_churn['email_open_count'] / (df_churn['days_since_signup'] / 30 + 1)
df_churn['avg_review_score_filled'] = df_churn['avg_review_score'].fillna(df_churn['avg_review_score'].median())

CHURN_FEATURES = [
    'order_count', 'total_spend', 'avg_order_value', 'coupon_uses', 'coupon_rate', 'purchase_frequency',
    'login_days_30', 'avg_session_duration', 'favorites_count', 'abandoned_carts', 'review_count',
    'email_open_count', 'is_subscriber', 'repurchased_after_low_review', 'days_since_signup',
    'purchase_span_days', 'email_open_per_month', 'main_category_enc', 'main_device_enc',
    'traffic_channel_enc', 'purchase_hour_zone_enc', 'segment_label_enc', 'sns_sentiment_score',
    'competitor_price_diff', 'brand_trend_exposure', 'avg_review_score_filled'
]
X = df_churn[CHURN_FEATURES]
y = df_churn['churned']
print(f"✅ 離脱予測のデータ準備完了: {X.shape}")

# %%
# ============================================================
# 【セル2】離脱予測モデル学習 (LightGBM Binary)
# ============================================================
print("--- 離脱予測モデルの学習 ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_dataset = lgb.Dataset(X_train, label=y_train)
valid_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)

params_churn = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.05, 'is_unbalance': True,
    'random_state': 42, 'verbose': -1, 'n_jobs': -1
}

model_churn = lgb.train(
    params_churn, train_set=train_dataset, num_boost_round=500,
    valid_sets=[train_dataset, valid_dataset], valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
)

y_pred_proba = model_churn.predict(X_test, num_iteration=model_churn.best_iteration)
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"✅ 離脱予測 AUC-ROC スコア: {auc_score:.4f}")

# 全顧客にスコアを付与
df_churn['churn_probability'] = model_churn.predict(df_churn[CHURN_FEATURES], num_iteration=model_churn.best_iteration)

# %%
# ============================================================
# 【セル3】LTV予測モデル学習 (LightGBM Regressor)
# ============================================================
print("--- LTV予測モデルの学習 ---")
LTV_FEATURES = CHURN_FEATURES.copy()
LTV_FEATURES.remove('total_spend')
LTV_FEATURES.append('age')

X_ltv = df_churn[LTV_FEATURES]
y_ltv = df_churn['total_spend']

X_ltv_train, X_ltv_test, y_ltv_train, y_ltv_test = train_test_split(X_ltv, y_ltv, test_size=0.2, random_state=42)
train_ltv = lgb.Dataset(X_ltv_train, label=y_ltv_train)
valid_ltv = lgb.Dataset(X_ltv_test, label=y_ltv_test, reference=train_ltv)

params_ltv = {
    'objective': 'regression', 'metric': 'rmse', 'num_leaves': 127,
    'learning_rate': 0.05, 'lambda_l2': 1.0, 'random_state': 42, 'verbose': -1, 'n_jobs': -1
}

model_ltv = lgb.train(
    params_ltv, train_set=train_ltv, num_boost_round=500,
    valid_sets=[train_ltv, valid_ltv], valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
)

y_ltv_pred = np.clip(model_ltv.predict(X_ltv_test, num_iteration=model_ltv.best_iteration), 0, None)
r2 = r2_score(y_ltv_test, y_ltv_pred)
rmse = np.sqrt(mean_squared_error(y_ltv_test, y_ltv_pred))
print(f"✅ LTV予測 R2スコア: {r2:.4f}, RMSE: ¥{rmse:,.0f}")

# 全顧客にLTV予測を付与
df_churn['predicted_ltv'] = np.clip(model_ltv.predict(df_churn[LTV_FEATURES], num_iteration=model_ltv.best_iteration), 0, None)

# %%
# ============================================================
# 【セル4】クーポンROI最適化シミュレーション（強化版）
# ============================================================
print("--- クーポンROI最適化シミュレーション（強化版） ---")

# セグメント別のuplift係数（ロイヤルは低い、離反・心理バグは高い）
SEG_UPLIFT_MULTIPLIER = {
    'Seg1_ロイヤル顧客': 0.3,    # 既に買っているのでuplift小さい
    'Seg2_一般顧客': 0.5,
    'Seg3_離反顧客': 0.8,        # 離反者はクーポンに反応しやすい
    'Seg4_見込み優良顧客': 0.6,
    'Seg5_見込み一般顧客': 0.5,
    'Seg6_休眠顧客': 0.7,
    'Seg7_潜在認知顧客': 0.4,
    'Seg8_深夜葛藤層': 0.9,      # 迷っているのでクーポンで後押し
    'Seg9_低評価中毒層': 0.6,
}

results = []
for seg_name in df['segment_label'].unique():
    seg_d = df[df['segment_label'] == seg_name]
    avg_spend = seg_d['total_spend'].mean()
    n_cust = len(seg_d)
    uplift_mult = SEG_UPLIFT_MULTIPLIER.get(seg_name, 0.5)

    for discount in DISCOUNT_RATES:
        uplift = uplift_mult * np.sqrt(discount)
        inc_rev = uplift * COUPON_USAGE_RATE * avg_spend * n_cust
        coupon_cost = COUPON_USAGE_RATE * discount * avg_spend * n_cust
        delivery_cost = n_cust * MAIL_COST_PER_SEND
        total_cost = coupon_cost + delivery_cost
        roi = ((inc_rev - total_cost) / total_cost * 100 if total_cost > 0 else 0)
        # 損益分岐uplift
        breakeven_uplift = (COUPON_USAGE_RATE * discount * avg_spend + MAIL_COST_PER_SEND) / (COUPON_USAGE_RATE * avg_spend) if avg_spend > 0 else 0
        results.append({
            'segment': seg_name,
            'discount_rate': discount,
            'discount_pct': f"{int(discount*100)}%",
            'n_customers': n_cust,
            'incremental_revenue': inc_rev,
            'total_cost': total_cost,
            'roi_pct': roi,
            'breakeven_uplift': breakeven_uplift
        })

sim_df = pd.DataFrame(results)
best_row = sim_df.loc[sim_df['roi_pct'].idxmax()]
print(f"✅ 最適クーポン: {best_row['segment']} に {best_row['discount_pct']} 割引 (ROI: {best_row['roi_pct']:.1f}%)")

# %%
# ============================================================
# 【セル4b】📊 Plotly版 ROI感度分析ヒートマップ
# ============================================================
print("--- 📊 Plotly: ROI感度分析ヒートマップ ---")

heatmap_data = sim_df.pivot_table(values='roi_pct', index='segment', columns='discount_pct')

fig_roi = px.imshow(
    heatmap_data,
    text_auto='.0f',
    color_continuous_scale='RdYlGn',
    labels=dict(x="割引率", y="セグメント", color="ROI (%)"),
    title='📊 セグメント × 割引率 ROI感度分析ヒートマップ',
    template='plotly_dark',
    aspect='auto'
)
fig_roi.update_layout(height=500)
fig_roi.show()

# %%
# ============================================================
# 【セル5】💰 離脱防止キャンペーン 収益シミュレーション
# ============================================================
print("=" * 70)
print("💰 離脱防止キャンペーン 収益シミュレーション")
print("=" * 70)

# 高リスク離脱顧客の抽出
high_risk = df_churn[df_churn['churn_probability'] >= CHURN_THRESHOLD].copy()
print(f"\n📌 高リスク離脱顧客数: {len(high_risk):,}人 (離脱確率 ≥ {CHURN_THRESHOLD*100:.0f}%)")
print(f"   平均予測LTV: ¥{high_risk['predicted_ltv'].mean():,.0f}")

# セグメント別の収益シミュレーション
print("\n--- セグメント別 施策シミュレーション ---")
scenarios = {
    '悲観': RETENTION_RATE_EMAIL,
    '基準': RETENTION_RATE_COUPON,
    '楽観': RETENTION_RATE_PERSONALIZED
}

roi_results = []
for seg_name in sorted(high_risk['segment_label'].unique()):
    seg_hr = high_risk[high_risk['segment_label'] == seg_name]
    n = len(seg_hr)
    avg_ltv = seg_hr['predicted_ltv'].mean()

    for scenario_name, success_rate in scenarios.items():
        # 最適割引率を取得
        seg_sim = sim_df[sim_df['segment'] == seg_name]
        best_discount = seg_sim.loc[seg_sim['roi_pct'].idxmax(), 'discount_rate'] if len(seg_sim) > 0 else 0.10

        recovered_customers = n * success_rate
        recovered_revenue = recovered_customers * avg_ltv * (1 - best_discount)
        campaign_cost = n * MAIL_COST_PER_SEND + recovered_customers * best_discount * avg_ltv
        saved_cac = recovered_customers * CAC  # 新規獲得コストの節約
        net_profit = recovered_revenue - campaign_cost + saved_cac
        roi = (net_profit / campaign_cost * 100) if campaign_cost > 0 else 0

        roi_results.append({
            'セグメント': seg_name,
            'シナリオ': scenario_name,
            '対象人数': n,
            '平均LTV': avg_ltv,
            '復帰見込人数': int(recovered_customers),
            '回収見込売上': recovered_revenue,
            '施策コスト': campaign_cost,
            'CAC節約額': saved_cac,
            '純利益': net_profit,
            'ROI(%)': roi
        })

roi_df = pd.DataFrame(roi_results)

# 基準シナリオの結果を表示
print("\n📊 基準シナリオ（クーポン施策・成功率15%）の結果:")
baseline = roi_df[roi_df['シナリオ'] == '基準'].copy()
baseline['回収見込売上'] = baseline['回収見込売上'].apply(lambda x: f"¥{x:,.0f}")
baseline['施策コスト'] = baseline['施策コスト'].apply(lambda x: f"¥{x:,.0f}")
baseline['純利益'] = baseline['純利益'].apply(lambda x: f"¥{x:,.0f}")
baseline['ROI(%)'] = baseline['ROI(%)'].apply(lambda x: f"{x:.1f}%")
display(baseline[['セグメント', '対象人数', '復帰見込人数', '回収見込売上', '施策コスト', '純利益', 'ROI(%)']])

# 3シナリオの総合計
print("\n📊 3シナリオ比較（全セグメント合計）:")
scenario_summary = roi_df.groupby('シナリオ').agg(
    合計対象人数=('対象人数', 'sum'),
    合計回収売上=('回収見込売上', 'sum'),
    合計施策コスト=('施策コスト', 'sum'),
    合計純利益=('純利益', 'sum')
).round(0)
scenario_summary['ROI(%)'] = ((scenario_summary['合計純利益'] / scenario_summary['合計施策コスト']) * 100).round(1)
for col in ['合計回収売上', '合計施策コスト', '合計純利益']:
    scenario_summary[col] = scenario_summary[col].apply(lambda x: f"¥{x:,.0f}")
display(scenario_summary)

# %%
# ============================================================
# 【セル5b】📊 Plotly版 施策インパクト ウォーターフォール図
# ============================================================
print("--- 📊 Plotly: 施策インパクト ウォーターフォール図 ---")

# 基準シナリオで各セグメントの純利益をウォーターフォールで表示
baseline_raw = roi_df[roi_df['シナリオ'] == '基準'].sort_values('純利益', ascending=False)

waterfall_labels = baseline_raw['セグメント'].tolist() + ['合計']
waterfall_values = baseline_raw['純利益'].tolist()
waterfall_measures = ['relative'] * len(baseline_raw) + ['total']

fig_waterfall = go.Figure(go.Waterfall(
    name="純利益",
    orientation="v",
    measure=waterfall_measures,
    x=waterfall_labels,
    y=waterfall_values + [0],  # 合計は自動計算される
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    increasing={"marker": {"color": "#34d399"}},
    decreasing={"marker": {"color": "#ef4444"}},
    totals={"marker": {"color": "#38bdf8"}}
))
fig_waterfall.update_layout(
    title="📊 施策インパクト ウォーターフォール図（基準シナリオ）",
    yaxis_title="純利益（円）",
    template='plotly_dark',
    height=500
)
fig_waterfall.show()

# %%
# ============================================================
# 【セル6】📋 アクションプラン（施策実行計画表）
# ============================================================
print("=" * 70)
print("📋 アクションプラン — 施策実行計画表")
print("=" * 70)

# 基準シナリオのROI結果を使ってアクションプランを生成
baseline_plan = roi_df[roi_df['シナリオ'] == '基準'].copy()

# 施策の定義
ACTION_MAP = {
    'Seg1_ロイヤル顧客': ('VIP限定先行販売・ポイント倍増', '来月'),
    'Seg2_一般顧客': ('カテゴリ別レコメンドメール', '今月中'),
    'Seg3_離反顧客': ('カムバックメール + 限定クーポン', '来週'),
    'Seg4_見込み優良顧客': ('無料サンプル + 初回割引', '今月中'),
    'Seg5_見込み一般顧客': ('メルマガ最適化 + A/Bテスト', '来月'),
    'Seg6_休眠顧客': ('Win-backシナリオメール（3段階）', '来週'),
    'Seg7_潜在認知顧客': ('SNS広告リターゲティング', '来月'),
    'Seg8_深夜葛藤層': ('深夜限定タイマークーポン', '即時'),
    'Seg9_低評価中毒層': ('品質改善報告メール + 特別クーポン', '今月中'),
}

action_rows = []
for _, row in baseline_plan.iterrows():
    seg = row['セグメント']
    action, timing = ACTION_MAP.get(seg, ('個別施策検討', '要検討'))
    roi_val = row['ROI(%)']
    # 優先度
    if roi_val >= 200:
        priority = '⭐⭐⭐ 最優先'
    elif roi_val >= 100:
        priority = '⭐⭐ 高'
    elif roi_val >= 0:
        priority = '⭐ 中'
    else:
        priority = '— 低'

    action_rows.append({
        '優先度': priority,
        'セグメント': seg,
        '推奨施策': action,
        '実施時期': timing,
        '対象人数': row['対象人数'],
        '想定コスト': f"¥{row['施策コスト']:,.0f}",
        '想定純利益': f"¥{row['純利益']:,.0f}",
        '想定ROI': f"{roi_val:.1f}%"
    })

action_df = pd.DataFrame(action_rows)
action_df = action_df.sort_values('優先度')
display(action_df)

print("\n🎯 推奨: 上位3施策を即時実行し、2週間後にA/Bテスト結果を確認")

# %%
# ============================================================
# 【セル7】AIメッセージ自動生成 (OpenAI API モック版)
# ============================================================
print("--- AIパーソナライズメッセージ自動生成 ---")
def generate_mock_message(segment_name, discount_pct, main_category):
    mock_data = {
        'ロイヤル顧客': (f'VIPのあなたへ、特別なご案内です', f'いつもご愛顧いただきありがとうございます。感謝を込めて、{main_category}カテゴリに{discount_pct}OFFの限定クーポンをご用意しました。', '今すぐ特典を確認する'),
        '離反顧客': (f'お久しぶりです。また会いたかったです', f'最近お顔を見ていないのが寂しく思っておりました。ぜひ{discount_pct}OFFでまたのご利用をお待ちしています。', 'カムバッククーポンを使う'),
        '深夜葛藤層': (f'迷っているあなたへ。今夜だけの決断を', f'ずっと気になっていた{main_category}商品に、今夜限り{discount_pct}OFFクーポンをご用意しました。後悔しないために。', '今すぐカートを確認する'),
        '低評価中毒層': (f'ご不満の声を受け、改善しました', f'先日のご不満にお応えし、{main_category}カテゴリの品質を改善いたしました。感謝を込めて{discount_pct}OFFクーポンをどうぞ。', '改善内容を確認する'),
    }
    subj, body, cta = mock_data.get(segment_name, (
        f'{segment_name}の皆様へ特別なご案内',
        f'{main_category}カテゴリの商品が{discount_pct}OFFになるクーポンをお届けします。',
        '今すぐ確認する'
    ))
    return {'subject': subj, 'body': body, 'cta': cta}

best_discounts = sim_df.sort_values('roi_pct', ascending=False).groupby('segment')['discount_pct'].first().to_dict()
seg_main_cat = df.groupby('segment_label')['main_category'].agg(lambda x: x.value_counts().index[0]).to_dict()

messages = []
for seg in sorted(df['segment_label'].unique()):
    seg_short = seg.split('_')[1]
    res = generate_mock_message(seg_short, best_discounts.get(seg, '10%'), seg_main_cat.get(seg, 'コスメ'))
    messages.append({'セグメント': seg, '件名': res['subject'], '本文': res['body']})

messages_df = pd.DataFrame(messages)
display(messages_df)
print("✅ 全セグメントのメッセージ生成完了")

# %%
# ============================================================
# 【セル8】パイプライン関数化 & 完了
# ============================================================
print("=" * 70)
print("🎉 全4ノートブックの分析完了！")
print("=" * 70)
print("""
  ─────────────────────────────────────────────────
  📖 使用したマーケティング理論:
  ─────────────────────────────────────────────────""")
theories = [
    ("佐藤尚之", "『ファンベース』", "上位20%ファン層の定量分析"),
    ("森岡毅",   "『確率思考の戦略論』", "NBDモデルによる購買確率シミュレーション"),
    ("西口一希", "『顧客起点マーケティング』", "行動データによる9セグ動的マッピング"),
    ("松本健太郎","『人は悪魔に熱狂する』", "深夜葛藤層・低評価中毒層の心理バグ抽出"),
]
for author, book, impl in theories:
    print(f"  📖 {author:<10} {book:<25} → {impl}")

print(f"\n  🏆 ポートフォリオ完成！")
print(f"     CRMマーケター × データサイエンティスト転職用")
print(f"     超一気通貫型 CRM戦略パッケージ（4ノートブック構成）")
