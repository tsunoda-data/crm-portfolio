"""
Segment transition tracking for Dynamics analysis (Phase 5).
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransitionReport:
    """Segment transition report."""
    transition_matrix: pd.DataFrame
    upgraded: List[str]
    downgraded: List[str]
    unchanged: int
    new_customers: int
    churned_customers: int
    alerts: List[Dict[str, str]]


def track_transitions(current_df: pd.DataFrame, previous_df: pd.DataFrame, key_col: str = 'customer_id', seg_col: str = 'segment_label') -> TransitionReport:
    """Compares segments between two time periods."""
    logger.info("Tracking segment transitions...")
    
    merged = pd.merge(
        previous_df[[key_col, seg_col]].rename(columns={seg_col: 'prev_seg'}),
        current_df[[key_col, seg_col]].rename(columns={seg_col: 'curr_seg'}),
        on=key_col,
        how='outer'
    )
    
    merged['prev_seg'] = merged['prev_seg'].fillna('New')
    merged['curr_seg'] = merged['curr_seg'].fillna('Churned')
    
    trans_matrix = pd.crosstab(merged['prev_seg'], merged['curr_seg'])
    
    new_custs = len(merged[merged['prev_seg'] == 'New'])
    churned_custs = len(merged[merged['curr_seg'] == 'Churned'])
    
    both = merged[(merged['prev_seg'] != 'New') & (merged['curr_seg'] != 'Churned')]
    unchanged = len(both[both['prev_seg'] == both['curr_seg']])
    
    good_keywords = ['ロイヤル', '優良']
    bad_keywords = ['離反', '休眠', '低評価']
    
    def _is_upgrade(p, c):
        p_good = any(k in p for k in good_keywords)
        c_good = any(k in c for k in good_keywords)
        p_bad = any(k in p for k in bad_keywords)
        c_bad = any(k in c for k in bad_keywords)
        return (not p_good and c_good) or (p_bad and not c_bad)
        
    def _is_downgrade(p, c):
        p_good = any(k in p for k in good_keywords)
        c_good = any(k in c for k in good_keywords)
        p_bad = any(k in p for k in bad_keywords)
        c_bad = any(k in c for k in bad_keywords)
        return (p_good and not c_good) or (not p_bad and c_bad)
        
    upgraded = both[both.apply(lambda x: _is_upgrade(x['prev_seg'], x['curr_seg']), axis=1)][key_col].tolist()
    downgraded = both[both.apply(lambda x: _is_downgrade(x['prev_seg'], x['curr_seg']), axis=1)][key_col].tolist()
    
    alerts = detect_urgent_transitions(merged)
    
    logger.info(f"Transition tracked: {new_custs} new, {churned_custs} churned, {len(upgraded)} upgraded, {len(downgraded)} downgraded.")
    
    return TransitionReport(
        transition_matrix=trans_matrix,
        upgraded=upgraded,
        downgraded=downgraded,
        unchanged=unchanged,
        new_customers=new_custs,
        churned_customers=churned_custs,
        alerts=alerts
    )


def detect_urgent_transitions(transitions_df: pd.DataFrame) -> List[Dict[str, str]]:
    """Flags high-value customers who moved to negative segments."""
    alerts = []
    if 'prev_seg' not in transitions_df.columns or 'curr_seg' not in transitions_df.columns:
        return alerts
        
    critical_drops = transitions_df[
        (transitions_df['prev_seg'].str.contains('ロイヤル', na=False)) & 
        (transitions_df['curr_seg'].str.contains('離反|休眠', na=False, regex=True))
    ]
    
    for _, row in critical_drops.iterrows():
        alerts.append({
            "customer_id": str(row['customer_id']),
            "message": f"Critical Drop: {row['prev_seg']} -> {row['curr_seg']}"
        })
    return alerts


def generate_transition_summary(report: TransitionReport, run_date: str, output_dir: Path) -> None:
    """Saves transition matrix and alerts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    matrix_path = output_dir / f"transition_matrix_{run_date}.csv"
    report.transition_matrix.to_csv(matrix_path)
    
    summary = {
        "run_date": run_date,
        "new_customers": report.new_customers,
        "churned_customers": report.churned_customers,
        "unchanged": report.unchanged,
        "upgraded_count": len(report.upgraded),
        "downgraded_count": len(report.downgraded),
        "alerts_count": len(report.alerts),
        "alerts": report.alerts[:100]
    }
    
    json_path = output_dir / f"transition_summary_{run_date}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Transition summary saved to {output_dir}")


def build_transition_history(data_dir: Path, n_periods: int = 6) -> pd.DataFrame:
    """Reads historical transition reports and builds a trend DataFrame."""
    logger.info(f"Building transition history from {data_dir} for up to {n_periods} periods.")
    history_records = []
    
    summary_files = sorted(data_dir.glob("transition_summary_*.json"))
    
    for fpath in summary_files[-n_periods:]:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                history_records.append({
                    "run_date": data.get("run_date"),
                    "new_customers": data.get("new_customers", 0),
                    "churned_customers": data.get("churned_customers", 0),
                    "upgraded": data.get("upgraded_count", 0),
                    "downgraded": data.get("downgraded_count", 0)
                })
        except Exception as e:
            logger.warning(f"Failed to read {fpath}: {e}")
            
    df = pd.DataFrame(history_records)
    return df
