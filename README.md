# 🎯 CRM運用パイプライン — 超一気通貫型 CRM戦略パッケージ

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsunoda-data/crm-portfolio/blob/main/crm_portfolio.ipynb)
[![Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen)]()

> CRMマーケター × データサイエンティスト × ビジネスストラテジスト

## 📊 プロジェクト概要

10,000人の合成CRMデータを用いて、**マーケティング理論 × 機械学習 × 運用設計**を統合した一気通貫のCRM戦略パッケージです。

単なる分析ではなく、**日々運用できるバッチパイプライン**として設計:
- ✅ CLI一本で全ステップ実行可能
- ✅ データ品質モニタリング（スキーマ検証・欠損率・値域チェック）
- ✅ モデルライフサイクル管理（バージョニング・ドリフト検知・自動再学習）
- ✅ A/Bテストフレームワーク（検出力分析・多重検定補正・効果測定）
- ✅ フィードバックループ（想定ROI vs 実績ROI追跡）
- ✅ セグメント動的更新（遷移トラッキング・緊急アラート）

### 💰 期待されるROIインパクト
- 離脱防止キャンペーンによる年間回収見込: **¥1,500万〜¥4,000万**（3シナリオで算出）
- セグメント別最適クーポンROI: **145%〜320%**

---

## 🏗️ プロジェクト構成

```
crm-portfolio/
├── crm_portfolio.ipynb              ← 🚀 ランチャー (Colab用)
├── run_pipeline.py                  ← 🔧 CLIエントリポイント (運用用)
├── Makefile                         ← make train, make test, etc.
├── config/
│   ├── pipeline_config.yaml         ← 全設定 (ビジネスパラメータ・モデル設定)
│   └── data_quality_rules.yaml      ← データ品質ルール
├── src/
│   ├── config.py                    ← 設定ローダー
│   ├── pipeline/
│   │   ├── ingest.py                ← データ取り込み & 合成データ生成
│   │   ├── features.py              ← 特徴量エンジニアリング
│   │   ├── segment.py               ← K-Means → 9セグマッピング
│   │   ├── score.py                 ← 離脱/LTVスコアリング
│   │   └── segment_transition.py    ← セグメント遷移トラッキング
│   ├── models/
│   │   ├── train.py                 ← LightGBM モデル学習
│   │   ├── evaluate.py              ← モデル評価 (AUC/R²/Feature Importance)
│   │   ├── drift.py                 ← PSIドリフト検知
│   │   └── registry.py              ← モデルバージョニング・ロールバック
│   ├── campaigns/
│   │   ├── ab_test.py               ← A/Bテストフレームワーク
│   │   ├── experiment.py            ← 合成キャンペーン実績データ生成
│   │   ├── export.py                ← MAツール向けリスト出力
│   │   └── feedback.py              ← フィードバックループ
│   ├── quality/
│   │   └── data_quality.py          ← データ品質モニタリング
│   └── reporting/
│       └── roi_tracker.py           ← 想定ROI vs 実績ROI追跡
├── tests/                           ← 26テスト
├── docs/
│   ├── governance.md                ← データガバナンス設計書
│   └── dashboard_spec.md            ← 経営ダッシュボード仕様書
├── scripts/                         ← 元のNotebook用スクリプト
└── models/ & data/ & logs/          ← 自動生成ディレクトリ
```

---

## 🚀 実行方法

### 方法1: Google Colab（ポートフォリオ閲覧用）
上の **「Open in Colab」バッジ** をクリック → 全セルを順に実行

### 方法2: CLI（運用モード）
```bash
# セットアップ
git clone https://github.com/tsunoda-data/crm-portfolio.git
cd crm-portfolio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 全ステップ実行
python run_pipeline.py

# 特定ステップのみ
python run_pipeline.py --steps ingest,quality_check,features

# 日付指定
python run_pipeline.py --date 2026-08-21

# テスト
make test
```

### Makefile コマンド一覧
```bash
make help       # コマンド一覧
make install    # 依存ライブラリインストール
make all        # 全ステップ実行
make train      # データ生成〜モデル学習
make test       # テスト実行 (26テスト)
make clean      # 中間ファイル削除
```

---

## 🛠️ 使用技術スタック

| カテゴリ | 技術 |
|---------|------|
| **言語** | Python 3.10+ |
| **データ操作** | pandas, numpy, pyarrow |
| **可視化** | matplotlib, seaborn, **Plotly**（インタラクティブ） |
| **機械学習** | scikit-learn, **LightGBM** |
| **統計** | scipy (NBD, PSI, χ²検定, z検定, 検出力分析) |
| **設定管理** | YAML |
| **テスト** | pytest (26テスト) |
| **実行環境** | Google Colab / ローカル CLI |
| **クラウド設計** | GCP (BigQuery + GCS) — 設計書準備済 |

---

## 📚 理論的背景

| 著者 | 書籍 | 本プロジェクトでの活用 |
|------|------|---------------------|
| 佐藤尚之 | 『ファンベース』 | 上位20%ファン層の定量分析（パレート分析） |
| 森岡毅 | 『確率思考の戦略論』 | NBDモデルによる購買確率シミュレーション |
| 西口一希 | 『顧客起点マーケティング』 | 行動データによる9セグメント動的マッピング |
| 松本健太郎 | 『人は悪魔に熱狂する』 | 深夜葛藤層・低評価中毒層の心理バグ抽出 |

---

## 📄 ライセンス

MIT License
