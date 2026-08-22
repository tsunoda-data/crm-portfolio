"""
Campaign list export utilities (Phase 3).
"""
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def generate_campaign_list(df: pd.DataFrame, segment_actions: Dict[str, Tuple[str, float, str]], cfg: dict) -> pd.DataFrame:
    """
    Creates MA-tool-ready CSV data.
    segment_actions maps segment_label to (action_name, discount_rate, timing).
    """
    logger.info("Generating campaign list...")
    if 'segment_label' not in df.columns:
        raise ValueError("DataFrame must contain 'segment_label' column.")
        
    campaign_data = []
    
    for _, row in df.iterrows():
        seg = row['segment_label']
        action = segment_actions.get(seg)
        if not action:
            continue
            
        action_name, discount_rate, timing = action
        
        msg_subject = f"【特別なお知らせ】{action_name}のご案内"
        msg_body = f"いつもご利用ありがとうございます。今回限りの{int(discount_rate * 100)}%OFFクーポンをお届けします。"
        
        priority = 1 if '離反' in seg else 2
        
        campaign_data.append({
            'customer_id': row.get('customer_id', 'unknown'),
            'segment': seg,
            'action': action_name,
            'discount_rate': discount_rate,
            'timing': timing,
            'message_subject': msg_subject,
            'message_body': msg_body,
            'priority': priority
        })
        
    campaign_df = pd.DataFrame(campaign_data)
    logger.info(f"Generated campaign list with {len(campaign_df)} target customers.")
    return campaign_df


def export_to_csv(campaign_df: pd.DataFrame, output_path: Path) -> None:
    """Saves DataFrame as CSV with proper encoding for MA tools."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Exported campaign list to {output_path}")


def export_to_bigquery(campaign_df: pd.DataFrame, table_name: str, cfg: dict) -> None:
    """Stub for exporting campaign list to BigQuery."""
    project_id = cfg.get('gcp', {}).get('project_id', 'unknown_project')
    dataset = cfg.get('gcp', {}).get('dataset', 'unknown_dataset')
    full_table_id = f"{project_id}.{dataset}.{table_name}"
    
    logger.info(f"[BQ STUB] Would export {len(campaign_df)} rows to BigQuery table {full_table_id}")
    logger.info(f"[BQ STUB] Schema: {list(campaign_df.columns)}")
