"""
Synthetic campaign outcome data generator (Phase 4).
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def simulate_campaign_outcomes(df: pd.DataFrame, campaign_config: dict, cfg: dict) -> pd.DataFrame:
    """Simulates campaign outcomes (open, use, purchase) for scored customers."""
    logger.info("Simulating campaign outcomes...")
    df_out = df.copy()
    np.random.seed(cfg['synthetic']['random_seed'])
    
    required_cols = ['customer_id', 'experiment_group', 'segment_label']
    for col in required_cols:
        if col not in df_out.columns:
            if col == 'experiment_group':
                df_out['experiment_group'] = 'Treatment'
            elif col == 'segment_label':
                df_out['segment_label'] = 'Unknown'
                
    baseline_open_rate = 0.20
    baseline_coupon_rate = cfg['business'].get('coupon_usage_rate', 0.40)
    baseline_purchase_rate = 0.10
    
    multipliers = cfg['business'].get('seg_uplift_multiplier', {})
    
    outcomes = []
    
    for _, row in df_out.iterrows():
        is_treatment = row['experiment_group'] == 'Treatment'
        seg = row['segment_label']
        mult = multipliers.get(seg, 0.5)
        
        email_sent = 1 if is_treatment else 0
        
        open_prob = baseline_open_rate * (1 + mult) if is_treatment else 0.0
        email_opened = 1 if np.random.rand() < open_prob else 0
        
        coupon_prob = baseline_coupon_rate * (1 + mult) if email_opened else 0.0
        coupon_used = 1 if np.random.rand() < coupon_prob else 0
        
        base_purchase = baseline_purchase_rate * (1 + (mult * 0.5))
        camp_purchase = 0.15 if coupon_used else (0.05 if email_opened else 0.0)
        
        prob_7d = min(1.0, base_purchase + camp_purchase)
        prob_14d = min(1.0, prob_7d + 0.05)
        prob_30d = min(1.0, prob_14d + 0.05)
        
        purchased_7d = 1 if np.random.rand() < prob_7d else 0
        purchased_14d = 1 if purchased_7d or np.random.rand() < (prob_14d - prob_7d) else 0
        purchased_30d = 1 if purchased_14d or np.random.rand() < (prob_30d - prob_14d) else 0
        
        purchase_amount = 0.0
        if purchased_30d:
            purchase_amount = np.random.lognormal(mean=8.5, sigma=0.5)
            purchase_amount = round(purchase_amount, -2)
            
        outcomes.append({
            'customer_id': row['customer_id'],
            'experiment_group': row['experiment_group'],
            'segment_label': seg,
            'email_sent': email_sent,
            'email_opened': email_opened,
            'coupon_used': coupon_used,
            'purchased_7d': purchased_7d,
            'purchased_14d': purchased_14d,
            'purchased_30d': purchased_30d,
            'purchase_amount': purchase_amount
        })
        
    res_df = pd.DataFrame(outcomes)
    logger.info(f"Simulation generated {len(res_df)} outcome records.")
    return res_df


def generate_multi_wave_outcomes(df: pd.DataFrame, cfg: dict, n_waves: int = 3) -> pd.DataFrame:
    """Generates monthly waves of campaign data."""
    logger.info(f"Generating multi-wave outcomes for {n_waves} waves...")
    all_waves = []
    
    for wave in range(1, n_waves + 1):
        wave_df = simulate_campaign_outcomes(df, {}, cfg)
        wave_df['wave'] = wave
        all_waves.append(wave_df)
        
    res = pd.concat(all_waves, ignore_index=True)
    logger.info(f"Multi-wave simulation complete. Total records: {len(res)}")
    return res
