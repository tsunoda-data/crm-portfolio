.PHONY: help install ingest train score test lint clean all

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## 依存ライブラリをインストール
	pip install -r requirements.txt

# ============================================================
# パイプライン実行
# ============================================================
ingest: ## データ取り込み + クレンジング
	python run_pipeline.py --steps ingest,quality_check

features: ## 特徴量エンジニアリング
	python run_pipeline.py --steps features

segment: ## セグメンテーション
	python run_pipeline.py --steps segment

train: ## モデル学習（離脱予測 + LTV予測）
	python run_pipeline.py --steps ingest,quality_check,features,segment,score

score: ## スコアリングのみ（学習済みモデル使用）
	python run_pipeline.py --steps score

export: ## キャンペーンリスト出力
	python run_pipeline.py --steps export

all: ## 全ステップ実行
	python run_pipeline.py

# ============================================================
# モデル管理
# ============================================================
drift: ## ドリフト検知を実行
	python -c "from src.models.drift import check_data_drift; print('drift check placeholder')"

# ============================================================
# テスト
# ============================================================
test: ## テストを実行
	python -m pytest tests/ -v --tb=short

test-quality: ## データ品質テストのみ
	python -m pytest tests/test_data_quality.py -v

test-ab: ## A/Bテストのみ
	python -m pytest tests/test_ab_test.py -v

# ============================================================
# ユーティリティ
# ============================================================
lint: ## コード品質チェック
	python -m flake8 src/ --max-line-length=120 --ignore=E501,W503

clean: ## 中間ファイルを削除
	rm -rf data/*.parquet logs/*.log models/*.pkl models/*.json __pycache__ src/__pycache__ src/**/__pycache__
