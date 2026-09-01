"""
CRM パイプライン CLI エントリポイント

使い方:
    python run_pipeline.py                          # 全ステップ実行
    python run_pipeline.py --steps ingest,score     # 特定ステップのみ
    python run_pipeline.py --date 2026-08-21        # 日付指定
    python run_pipeline.py --config path/to/cfg.yaml
    python run_pipeline.py --mode synthetic         # 合成データモード
    python run_pipeline.py --mode production        # 本番データモード
"""
import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

# ============================================================
# パス解決: src/models/ と models/ の両方に対応
# Cloud Shell や各種環境でのモジュール配置の差異を吸収する
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# src/models が存在しない場合は src/ を sys.path に追加して
# models/ を src.models として解決できるようにする
_src_models = PROJECT_ROOT / "src" / "models"
_root_models = PROJECT_ROOT / "models"
if not _src_models.exists() and _root_models.exists():
    # models/ が直下にある環境（Cloud Shell など）
    # src/ がなければ src/ を作って models/ へのシムを張る代わりに
    # src ディレクトリを sys.path に追加してインポートを解決
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _smart_import(src_path: str, fallback_path: str):
    """src.X.Y が失敗した場合に X.Y (srcなし) でフォールバックするimport helper"""
    import importlib
    try:
        return importlib.import_module(src_path)
    except ModuleNotFoundError:
        return importlib.import_module(fallback_path)

from src.config import load_config, reset_config_cache, get_pipeline_paths


def setup_logging(log_dir: Path, run_date: str) -> logging.Logger:
    """構造化ログの設定。ファイルとコンソールに出力。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{run_date}.log"

    logger = logging.getLogger("crm_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # ファイルハンドラ
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # コンソールハンドラ
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def run_step(step_name: str, cfg: dict, logger: logging.Logger, context: dict) -> dict:
    """個別ステップを実行。contextに中間結果を格納。"""

    if step_name == "ingest":
        from src.pipeline.ingest import generate_synthetic, cleanse, save_checkpoint
        logger.info("=" * 60)
        logger.info("STEP: ingest — データ取り込み & クレンジング")
        logger.info("=" * 60)
        df_raw = generate_synthetic(cfg)
        logger.info(f"生データ生成完了: {df_raw.shape}")
        df_clean = cleanse(df_raw, cfg)
        logger.info(f"クレンジング完了: {df_clean.shape}")
        save_checkpoint(df_clean, "cleaned", cfg)
        context["df_clean"] = df_clean

    elif step_name == "quality_check":
        from src.quality.data_quality import DataQualityChecker
        logger.info("=" * 60)
        logger.info("STEP: quality_check — データ品質チェック")
        logger.info("=" * 60)
        checker = DataQualityChecker(cfg)
        report = checker.run_all_checks(context["df_clean"])
        if not report.passed:
            logger.warning(f"データ品質チェック失敗: {len(report.issues)} 件の問題")
            for issue in report.issues:
                logger.warning(f"  ⚠️  {issue}")
        else:
            logger.info("✅ データ品質チェック全パス")
        context["quality_report"] = report

    elif step_name == "features":
        from src.pipeline.features import build_churn_features, build_rfm_features
        logger.info("=" * 60)
        logger.info("STEP: features — 特徴量エンジニアリング")
        logger.info("=" * 60)
        run_date = cfg["pipeline"]["run_date"]
        df = context["df_clean"]
        df = build_churn_features(df, run_date)
        df = build_rfm_features(df, run_date)
        context["df_features"] = df
        logger.info(f"特徴量生成完了: {df.shape[1]} カラム")

    elif step_name == "segment":
        from src.pipeline.segment import fit_segments, predict_segments, apply_psychology_segments, get_segment_summary
        from src.pipeline.ingest import save_checkpoint
        logger.info("=" * 60)
        logger.info("STEP: segment — セグメンテーション")
        logger.info("=" * 60)
        df = context["df_features"]
        # fit_segments は (kmeans, scaler, profile_scaled, thresholds) の4タプルを返す
        kmeans, scaler, profile_scaled, thresholds = fit_segments(df)
        # predict_segments で segment_label カラムをDataFrameに付与する
        df = predict_segments(df, kmeans, scaler, profile_scaled, thresholds)
        df = apply_psychology_segments(df)
        summary = get_segment_summary(df)
        logger.info(f"セグメント分布:\n{summary}")
        save_checkpoint(df, "segmented", cfg)
        context["df_segmented"] = df
        context["seg_artifacts"] = {"kmeans": kmeans, "scaler": scaler,
                                    "profile_scaled": profile_scaled, "thresholds": thresholds}

    elif step_name == "score":
        from src.pipeline.score import score_churn, score_ltv, classify_risk
        from src.pipeline.features import CHURN_FEATURE_COLUMNS, LTV_FEATURE_COLUMNS
        _reg_mod = _smart_import("src.models.registry", "models.registry")
        ModelRegistry = _reg_mod.ModelRegistry
        logger.info("=" * 60)
        logger.info("STEP: score — モデルスコアリング")
        logger.info("=" * 60)
        df = context["df_segmented"]

        registry = ModelRegistry(cfg["pipeline"]["model_dir"])

        # モデルが存在しない場合は学習から実行
        try:
            churn_model, churn_meta = registry.load_model("churn")
            ltv_model, ltv_meta = registry.load_model("ltv")
            logger.info(f"モデル読み込み: churn v{churn_meta['version']}, ltv v{ltv_meta['version']}")
        except FileNotFoundError:
            logger.info("学習済みモデルが見つかりません。新規学習を実行します。")
            _train_mod = _smart_import("src.models.train", "models.train")
            train_churn_model = _train_mod.train_churn_model
            train_ltv_model = _train_mod.train_ltv_model
            from src.pipeline.features import CHURN_FEATURE_COLUMNS, LTV_FEATURE_COLUMNS
            from sklearn.model_selection import train_test_split
            import pandas as pd

            df_purchasers = df[df["order_count"] > 0].copy()
            X = df_purchasers[CHURN_FEATURE_COLUMNS]
            y = df_purchasers["churned"]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            churn_model, churn_metrics = train_churn_model(X_train, y_train, X_val, y_val, cfg)
            registry.save_model(churn_model, "churn", churn_metrics, CHURN_FEATURE_COLUMNS, cfg["pipeline"]["run_date"])
            logger.info(f"離脱モデル学習完了: AUC={churn_metrics.get('auc', 'N/A')}")

            X_ltv = df_purchasers[LTV_FEATURE_COLUMNS]
            y_ltv = df_purchasers["total_spend"]
            Xt, Xv, yt, yv = train_test_split(X_ltv, y_ltv, test_size=0.2, random_state=42)
            ltv_model, ltv_metrics = train_ltv_model(Xt, yt, Xv, yv, cfg)
            registry.save_model(ltv_model, "ltv", ltv_metrics, LTV_FEATURE_COLUMNS, cfg["pipeline"]["run_date"])
            logger.info(f"LTVモデル学習完了: R2={ltv_metrics.get('r2', 'N/A')}")

        df = score_churn(df, churn_model, CHURN_FEATURE_COLUMNS)
        df = score_ltv(df, ltv_model, LTV_FEATURE_COLUMNS)
        threshold = cfg["business"]["churn_threshold"]
        df = classify_risk(df, threshold)
        context["df_scored"] = df
        logger.info(f"スコアリング完了: 高リスク顧客 {(df['risk_level'] == 'high').sum()} 人")

    elif step_name == "export":
        from src.campaigns.export import generate_campaign_list, export_to_csv
        from src.pipeline.ingest import save_checkpoint
        logger.info("=" * 60)
        logger.info("STEP: export — キャンペーンリスト出力")
        logger.info("=" * 60)
        df = context.get("df_scored", context.get("df_segmented"))

        # セグメント別アクション定義: {segment_label: (action_name, discount_rate, timing)}
        segment_actions = cfg.get("campaigns", {}).get("segment_actions") or {
            "Seg1_ロイヤル顧客":    ("VIP感謝キャンペーン",   0.05, "週次"),
            "Seg2_一般顧客":        ("リピート促進クーポン",   0.10, "月次"),
            "Seg3_離反顧客":        ("カムバックオファー",     0.20, "即時"),
            "Seg4_見込み優良顧客":  ("初回購入サポート",       0.15, "週次"),
            "Seg5_見込み一般顧客":  ("お試しクーポン",         0.10, "月次"),
            "Seg6_休眠顧客":        ("再活性化キャンペーン",   0.25, "即時"),
            "Seg7_潜在認知顧客":    ("ブランド体験クーポン",   0.10, "月次"),
            "Seg8_深夜葛藤層":      ("深夜限定フラッシュセール", 0.15, "深夜限定"),
            "Seg9_低評価中毒層":    ("品質改善お詫びオファー", 0.30, "即時"),
        }

        campaign_df = generate_campaign_list(df, segment_actions, cfg)
        output_path = Path(cfg["pipeline"]["data_dir"]) / f"campaign_list_{cfg['pipeline']['run_date']}.csv"
        export_to_csv(campaign_df, output_path)
        save_checkpoint(df, "scored_final", cfg)
        context["campaign_df"] = campaign_df
        logger.info(f"キャンペーンリスト出力完了: {len(campaign_df)} 件")

    else:
        logger.warning(f"未知のステップ: {step_name}")

    return context


def main():
    parser = argparse.ArgumentParser(description="CRM パイプライン")
    parser.add_argument("--config", type=str, default=None, help="設定ファイルパス")
    parser.add_argument("--date", type=str, default=None, help="実行日 (YYYY-MM-DD)")
    parser.add_argument("--steps", type=str, default=None, help="実行ステップ (カンマ区切り)")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "production"],
                        help="実行モード")
    args = parser.parse_args()

    # 設定読み込み
    reset_config_cache()
    cfg = load_config(args.config)

    # 日付の上書き
    if args.date:
        cfg["pipeline"]["run_date"] = args.date

    # ステップの決定
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
    else:
        steps = cfg["pipeline"]["steps"]

    # パス設定
    paths = get_pipeline_paths(cfg)

    # ログ設定
    run_date = cfg["pipeline"]["run_date"]
    logger = setup_logging(paths["log_dir"], run_date)

    logger.info("🚀 CRM パイプライン開始")
    logger.info(f"   実行日: {run_date}")
    logger.info(f"   モード: {args.mode}")
    logger.info(f"   ステップ: {steps}")
    logger.info(f"   設定: {args.config or 'デフォルト'}")

    start_time = time.time()
    context = {}

    try:
        for step in steps:
            step_start = time.time()
            context = run_step(step, cfg, logger, context)
            elapsed = time.time() - step_start
            logger.info(f"⏱️  {step} 完了 ({elapsed:.1f}秒)")

        total_elapsed = time.time() - start_time
        logger.info(f"🎉 パイプライン完了 (合計 {total_elapsed:.1f}秒)")

    except Exception as e:
        logger.error(f"❌ パイプラインエラー: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
