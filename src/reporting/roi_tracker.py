"""
ROI Tracker: 想定ROI vs 実績ROI の比較・追跡

フィードバックループの出口として、施策のROI実績を記録し、
仮定パラメータとの乖離を定量化する。
"""
import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("crm_pipeline.reporting.roi_tracker")


@dataclass
class ROIComparison:
    """1セグメントの想定 vs 実績ROI比較結果。"""
    segment: str
    predicted_roi: float
    actual_roi: float
    gap_pct: float            # (actual - predicted) / predicted * 100
    predicted_revenue: float
    actual_revenue: float
    predicted_cost: float
    actual_cost: float
    status: str               # "on_track" / "underperform" / "overperform"


@dataclass
class ROITrackingReport:
    """ROI追跡レポート全体。"""
    run_date: str
    campaign_name: str
    comparisons: list[ROIComparison]
    total_predicted_roi: float
    total_actual_roi: float
    total_gap_pct: float
    recommendations: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def compare_predicted_vs_actual(
    predicted_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    segment_col: str = "segment"
) -> list[ROIComparison]:
    """
    想定ROIと実績ROIを比較する。

    Parameters
    ----------
    predicted_df : DataFrame
        必須カラム: segment, predicted_revenue, predicted_cost, predicted_roi
    actual_df : DataFrame
        必須カラム: segment, actual_revenue, actual_cost, actual_roi

    Returns
    -------
    list[ROIComparison]
    """
    comparisons = []

    merged = pd.merge(predicted_df, actual_df, on=segment_col, how="outer", suffixes=("_pred", "_act"))

    for _, row in merged.iterrows():
        seg = row[segment_col]
        pred_roi = row.get("predicted_roi", 0) or row.get("roi_pred", 0)
        act_roi = row.get("actual_roi", 0) or row.get("roi_act", 0)
        pred_rev = row.get("predicted_revenue", 0) or row.get("revenue_pred", 0)
        act_rev = row.get("actual_revenue", 0) or row.get("revenue_act", 0)
        pred_cost = row.get("predicted_cost", 0) or row.get("cost_pred", 0)
        act_cost = row.get("actual_cost", 0) or row.get("cost_act", 0)

        # 乖離率
        if pred_roi != 0:
            gap = (act_roi - pred_roi) / abs(pred_roi) * 100
        else:
            gap = 0.0

        # ステータス判定
        if abs(gap) <= 15:
            status = "on_track"
        elif gap < -15:
            status = "underperform"
        else:
            status = "overperform"

        comparisons.append(ROIComparison(
            segment=seg,
            predicted_roi=round(pred_roi, 1),
            actual_roi=round(act_roi, 1),
            gap_pct=round(gap, 1),
            predicted_revenue=round(pred_rev, 0),
            actual_revenue=round(act_rev, 0),
            predicted_cost=round(pred_cost, 0),
            actual_cost=round(act_cost, 0),
            status=status,
        ))

    return comparisons


def generate_tracking_report(
    comparisons: list[ROIComparison],
    campaign_name: str,
    run_date: str,
) -> ROITrackingReport:
    """ROI追跡レポートを生成する。"""
    total_pred_rev = sum(c.predicted_revenue for c in comparisons)
    total_act_rev = sum(c.actual_revenue for c in comparisons)
    total_pred_cost = sum(c.predicted_cost for c in comparisons)
    total_act_cost = sum(c.actual_cost for c in comparisons)

    total_pred_roi = ((total_pred_rev - total_pred_cost) / total_pred_cost * 100) if total_pred_cost > 0 else 0
    total_act_roi = ((total_act_rev - total_act_cost) / total_act_cost * 100) if total_act_cost > 0 else 0
    total_gap = ((total_act_roi - total_pred_roi) / abs(total_pred_roi) * 100) if total_pred_roi != 0 else 0

    # レコメンデーション生成
    recommendations = []
    for c in comparisons:
        if c.status == "underperform":
            recommendations.append(
                f"⚠️ {c.segment}: 実績ROI({c.actual_roi}%)が想定({c.predicted_roi}%)を大幅に下回っています。"
                f" uplift係数の見直しを推奨します。"
            )
        elif c.status == "overperform":
            recommendations.append(
                f"📈 {c.segment}: 実績ROI({c.actual_roi}%)が想定({c.predicted_roi}%)を上回っています。"
                f" 予算増額を検討してください。"
            )

    if not recommendations:
        recommendations.append("✅ 全セグメントが想定通りのROIを達成しています。")

    return ROITrackingReport(
        run_date=run_date,
        campaign_name=campaign_name,
        comparisons=comparisons,
        total_predicted_roi=round(total_pred_roi, 1),
        total_actual_roi=round(total_act_roi, 1),
        total_gap_pct=round(total_gap, 1),
        recommendations=recommendations,
    )


def save_tracking_report(report: ROITrackingReport, output_dir: str | Path) -> Path:
    """ROI追跡レポートをJSONで保存する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"roi_tracking_{report.campaign_name}_{report.run_date}.json"
    filepath = output_dir / filename

    report_dict = asdict(report)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    logger.info(f"ROI追跡レポート保存: {filepath}")
    return filepath


def load_tracking_history(output_dir: str | Path, campaign_prefix: str = "") -> pd.DataFrame:
    """過去のROI追跡レポートを読み込み、トレンドDataFrameを返す。"""
    output_dir = Path(output_dir)
    records = []

    for f in sorted(output_dir.glob("roi_tracking_*.json")):
        if campaign_prefix and campaign_prefix not in f.name:
            continue
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        records.append({
            "run_date": data["run_date"],
            "campaign_name": data["campaign_name"],
            "predicted_roi": data["total_predicted_roi"],
            "actual_roi": data["total_actual_roi"],
            "gap_pct": data["total_gap_pct"],
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("run_date")
