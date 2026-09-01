#!/bin/bash
# =============================================================
# Cloud Shell セットアップスクリプト
# 使い方: bash setup_cloudshell.sh
# =============================================================
set -e
echo "======================================"
echo " CRM Pipeline — Cloud Shell セットアップ"
echo "======================================"

# ① 最新コードを pull
echo ""
echo "📥 [1/5] 最新コードを取得..."
git pull origin main 2>/dev/null || echo "  → (スキップ: pullエラー。ローカル状態で続行します)"

# ② src/models が存在しない場合、models/ からシンボリックリンクを作成
echo ""
echo "🔧 [2/5] モジュールパスを確認・修正..."
if [ ! -d "src" ]; then
  mkdir -p src
fi
if [ ! -f "src/__init__.py" ]; then
  touch src/__init__.py
  echo "  → src/__init__.py を作成しました"
fi
if [ ! -d "src/models" ] && [ -d "models" ]; then
  echo "  → src/models が存在しません。models/ をコピーします..."
  cp -r models src/models
  echo "  → src/models/ を作成しました（models/ からコピー）"
elif [ -d "src/models" ]; then
  echo "  → src/models/ は存在します ✅"
fi
if [ ! -f "src/models/__init__.py" ]; then
  touch src/models/__init__.py
  echo "  → src/models/__init__.py を作成しました"
fi

# ③ 必要なディレクトリを作成
echo ""
echo "📁 [3/5] 必要なディレクトリを作成..."
mkdir -p data models logs
echo "  → data/, models/, logs/ を作成しました ✅"

# ④ ライブラリをインストール
echo ""
echo "📦 [4/5] ライブラリをインストール..."
pip install -q -r requirements.txt
echo "  → インストール完了 ✅"

# ⑤ 動作確認
echo ""
echo "🧪 [5/5] 動作確認..."
PYTHONPATH=. python3 -c "
import sys
sys.path.insert(0, '.')
errors = []
checks = [
    ('src.config', 'src.config'),
    ('src.pipeline.ingest', 'src.pipeline.ingest'),
    ('src.pipeline.features', 'src.pipeline.features'),
    ('src.pipeline.segment', 'src.pipeline.segment'),
    ('src.pipeline.score', 'src.pipeline.score'),
    ('src.quality.data_quality', 'src.quality.data_quality'),
    ('src.campaigns.ab_test', 'src.campaigns.ab_test'),
]
for name, mod in checks:
    try:
        __import__(mod)
        print(f'  ✅ {name}')
    except Exception as e:
        print(f'  ⚠️  {name}: {e}')
        errors.append(name)

# models は src/ or root の両方を試す
import importlib
for mod_path, fallback in [('src.models.registry', 'models.registry')]:
    try:
        importlib.import_module(mod_path)
        print(f'  ✅ {mod_path}')
    except:
        try:
            importlib.import_module(fallback)
            print(f'  ✅ {fallback} (fallback)')
        except Exception as e:
            print(f'  ⚠️  {mod_path} / {fallback}: {e}')
            errors.append(mod_path)
"

echo ""
echo "======================================"
echo " セットアップ完了！"
echo ""
echo " 実行コマンド:"
echo "   PYTHONPATH=. python3 run_pipeline.py"
echo ""
echo " 特定ステップのみ実行:"
echo "   PYTHONPATH=. python3 run_pipeline.py --steps ingest,quality_check"
echo "======================================"
