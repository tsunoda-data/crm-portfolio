"""
Feedback loop processing for updating parameters from campaign results (Phase 4).
"""
import json
import logging
from pathlib import Path
from typing import Dict
import pandas as pd

logger = logging.getLogger(__name__)


def collect_campaign_results(outcomes_df: pd.DataFrame, campaign_df: pd.DataFrame) -> pd.DataFrame:
    """Joins campaign data with actual outcome data."""
    logger.info("Collecting campaign results...")
    if 'customer_id' not in outcomes_df.columns or 'customer_id' not in campaign_df.columns:
        raise ValueError("Both dataframes must contain 'customer_id'.")
        
    merged = pd.merge(campaign_df, outcomes_df, on='customer_id', how='left')
    logger.info(f"Merged results shape: {merged.shape}")
    return merged


def update_retention_rates(results_df: pd.DataFrame, current_rates: Dict[str, float]) -> Dict[str, float]:
    """Calculates actual retention rates from A/B test results and updates rates."""
    logger.info("Updating retention rates based on campaign results...")
    updated = current_rates.copy()
    
    if 'action' in results_df.columns and 'purchased_30d' in results_df.columns:
        action_rates = results_df.groupby('action')['purchased_30d'].mean().to_dict()
        for action, rate in action_rates.items():
            for k in updated.keys():
                if k.lower() in action.lower():
                    updated[k] = round(updated[k] * 0.5 + rate * 0.5, 3)
    
    logger.info(f"Updated retention rates: {updated}")
    return updated


def update_uplift_multipliers(results_df: pd.DataFrame, current_multipliers: Dict[str, float]) -> Dict[str, float]:
    """Updates SEG_UPLIFT_MULTIPLIER from measured uplift data."""
    logger.info("Updating segment uplift multipliers...")
    updated = current_multipliers.copy()
    
    if 'segment' in results_df.columns and 'purchased_30d' in results_df.columns and 'experiment_group' in results_df.columns:
        for seg in results_df['segment'].unique():
            seg_data = results_df[results_df['segment'] == seg]
            treat_mean = seg_data[seg_data['experiment_group'] == 'Treatment']['purchased_30d'].mean()
            ctrl_mean = seg_data[seg_data['experiment_group'] == 'Control']['purchased_30d'].mean()
            
            if pd.notna(treat_mean) and pd.notna(ctrl_mean) and ctrl_mean > 0:
                uplift = (treat_mean - ctrl_mean) / ctrl_mean
                uplift = max(0.1, min(1.5, uplift))
                
                if seg in updated:
                    updated[seg] = round(updated[seg] * 0.7 + uplift * 0.3, 2)
                    
    logger.info(f"Updated uplift multipliers: {updated}")
    return updated


def compare_predicted_vs_actual(predicted_roi_df: pd.DataFrame, actual_results_df: pd.DataFrame) -> pd.DataFrame:
    """Compares predicted ROI/values with actual results."""
    logger.info("Comparing predicted vs actual campaign outcomes...")
    
    merged = pd.merge(predicted_roi_df, actual_results_df, on='customer_id', how='inner')
    
    merged['actual_revenue'] = merged.get('purchase_amount', pd.Series(0)).fillna(0)
    
    # 簡易的にコスト計算
    email_sent = merged.get('email_sent', pd.Series(0))
    coupon_used = merged.get('coupon_used', pd.Series(0))
    actual_revenue = merged['actual_revenue']
    
    merged['actual_cost'] = email_sent * 3.0 + coupon_used * (actual_revenue * 0.1)
    
    actual_cost_nonzero = merged['actual_cost'].replace(0, 1)
    merged['actual_roi'] = (merged['actual_revenue'] - merged['actual_cost']) / actual_cost_nonzero
    
    if 'predicted_roi' in merged.columns:
        merged['roi_gap'] = merged['actual_roi'] - merged['predicted_roi']
    
    return merged


def save_feedback_report(comparison_df: pd.DataFrame, run_date: str, output_dir: Path) -> None:
    """Saves feedback report as Parquet + JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pq_path = output_dir / f"feedback_comparison_{run_date}.parquet"
    comparison_df.to_parquet(pq_path, index=False)
    
    summary = {
        "run_date": run_date,
        "total_customers": len(comparison_df),
        "avg_actual_roi": float(comparison_df.get('actual_roi', pd.Series([0.0])).mean()),
        "avg_roi_gap": float(comparison_df.get('roi_gap', pd.Series([0.0])).mean()),
        "total_actual_revenue": float(comparison_df.get('actual_revenue', pd.Series([0.0])).sum())
    }
    
    json_path = output_dir / f"feedback_summary_{run_date}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Feedback report saved to {output_dir}")
