"""
設定ローダー: config/pipeline_config.yaml を読み込み、全モジュールに提供する。
"""
import yaml
import os
from datetime import date
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


_CONFIG_CACHE: dict | None = None
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config(config_path: str | None = None) -> dict:
    """YAMLファイルを読み込んでdictで返す。一度読んだらキャッシュ。"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = _CONFIG_DIR / "pipeline_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # run_date の解決
    if cfg["pipeline"].get("run_date", "auto") == "auto":
        cfg["pipeline"]["run_date"] = date.today().isoformat()

    _CONFIG_CACHE = cfg
    return cfg


def reset_config_cache():
    """テスト用: キャッシュをクリアする。"""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def get_business_params(cfg: dict | None = None) -> dict:
    """ビジネスパラメータを取得。"""
    if cfg is None:
        cfg = load_config()
    return cfg["business"]


def get_model_config(model_name: str, cfg: dict | None = None) -> dict:
    """モデル固有の設定を取得。model_name: 'churn' or 'ltv'"""
    if cfg is None:
        cfg = load_config()
    return cfg["model"][model_name]


def get_pipeline_paths(cfg: dict | None = None) -> dict:
    """パイプラインで使うディレクトリパスを取得。存在しなければ作成。"""
    if cfg is None:
        cfg = load_config()
    paths = {
        "data_dir": Path(cfg["pipeline"]["data_dir"]),
        "log_dir": Path(cfg["pipeline"]["log_dir"]),
        "model_dir": Path(cfg["pipeline"]["model_dir"]),
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
