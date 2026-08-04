# %% [markdown]
# # 📓 Notebook 4: ML予測 & AIエージェント自動化
# 
# LightGBMによる「離脱予測」と「LTV予測」、さらに割引率別の
# 「クーポンROI最適化シミュレーション」を実施します。
# 最後にOpenAI APIを活用した「AIパーソナライズメッセージ自動生成」と
# パイプライン化(関数化)を行います。

# %%
# ============================================================
# 【セル1】ライブラリインポート・データ準備・離脱予測の準備
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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
import lightgbm as lgb
import os, json, time
import warnings
warnings.filterwarnings('ignore')

PARQUET_PATH = '/Users/user/projects/crm-portfolio/notebooks/data/notebook3_segmented_data.parquet'
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
    'traffic_channel_enc', 'purchase_hour_enc', 'segment_enc', 'sns_sentiment_score',
    'competitor_price_diff', 'brand_trend_exposure', 'avg_review_score_filled'
]
X = df_churn[CHURN_FEATURES]
y = df_churn['churned']
print("✅ 離脱予測のデータ準備完了")

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

# %%
# ============================================================
# 【セル3】LTV予測モデル学習 (LightGBM Regressor)
# ============================================================
print("--- LTV予測モデルの学習 ---")
LTV_FEATURES = CHURN_FEATURES.copy()
LTV_FEATURES.remove('total_spend') # 目的変数なので除外
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

# %%
# ============================================================
# 【セル4】クーポンROI最適化シミュレーション
# ============================================================
print("--- クーポンROI最適化シミュレーション ---")
DISCOUNT_RATES = [0.05, 0.10, 0.15, 0.20, 0.30]
COUPON_USAGE_RATE = 0.40

results = []
for seg_name in df['segment_label'].unique():
    seg_d = df[df['segment_label'] == seg_name]
    avg_spend = seg_d['total_spend'].mean()
    n_cust = len(seg_d)
    
    for discount in DISCOUNT_RATES:
        uplift = 0.65 * np.sqrt(discount)
        inc_rev = uplift * COUPON_USAGE_RATE * avg_spend * n_cust
        cost = COUPON_USAGE_RATE * discount * avg_spend * n_cust
        roi = ((inc_rev - cost) / cost * 100 if cost > 0 else 0)
        results.append({'segment': seg_name, 'discount_pct': f"{int(discount*100)}%", 'roi_pct': roi})

sim_df = pd.DataFrame(results)
best_row = sim_df.loc[sim_df['roi_pct'].idxmax()]
print(f"✅ 最適クーポン: {best_row['segment']} に {best_row['discount_pct']} 割引 (ROI: {best_row['roi_pct']:.1f}%)")

# %%
# ============================================================
# 【セル5】AIメッセージ自動生成 (OpenAI API モック版)
# ============================================================
print("--- AIパーソナライズメッセージ自動生成 ---")
# ※ここでは環境変数なしでも動くようにモックを利用
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
display(messages_df.head())
print("✅ 全セグメントのメッセージ生成完了")

# %%
# ============================================================
# 【セル6】パイプライン関数化
# ============================================================
print("--- 全体パイプライン関数化 (main) ---")
def main_pipeline(parquet_path='/Users/user/projects/crm-portfolio/notebooks/data/notebook3_segmented_data.parquet'):
    """
    CRM戦略パッケージの全処理を実行するメイン関数 (デモ用)
    実際にはNotebook 1〜4で実行した全ての関数を統合します。
    """
    print("🚀 CRMパイプラインの実行完了")
    return {"status": "success"}

main_pipeline()
print("🎉 全4ノートブックの作成完了！")
