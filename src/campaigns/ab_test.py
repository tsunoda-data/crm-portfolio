"""
A/B Test framework for CRM campaigns (Phase 3).
"""
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class UpliftResult:
    """A/B Test Uplift Result."""
    treatment_rate: float
    control_rate: float
    absolute_uplift: float
    relative_uplift: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    sample_sizes: Tuple[int, int]


def assign_groups(df: pd.DataFrame, control_ratio: float, salt: str = 'experiment_v1') -> pd.DataFrame:
    """Assigns customers to Treatment/Control using hash-based randomization."""
    logger.info(f"Assigning A/B test groups with control_ratio={control_ratio}")
    if 'customer_id' not in df.columns:
        raise ValueError("DataFrame must contain 'customer_id' column.")
    
    def _hash_customer(cid: str) -> float:
        # returns float between 0 and 1
        h = hashlib.md5(f"{cid}_{salt}".encode('utf-8')).hexdigest()
        return int(h, 16) / (16 ** 32)
    
    hash_vals = df['customer_id'].astype(str).apply(_hash_customer)
    df_out = df.copy()
    df_out['experiment_group'] = np.where(hash_vals < control_ratio, 'Control', 'Treatment')
    
    counts = df_out['experiment_group'].value_counts(normalize=True)
    logger.info(f"Assigned groups: {counts.to_dict()}")
    return df_out


def calculate_sample_size(baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Calculates minimum sample size per group for two-proportion z-test."""
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    
    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_bar = (p1 + p2) / 2
    
    n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (mde ** 2)
    return int(np.ceil(n))


def measure_uplift(treatment_df: pd.DataFrame, control_df: pd.DataFrame, metric_col: str, alpha: float = 0.05) -> UpliftResult:
    """Calculates uplift, runs statistical test (chi-squared for proportions)."""
    n_treat = len(treatment_df)
    n_ctrl = len(control_df)
    
    treat_success = treatment_df[metric_col].sum()
    ctrl_success = control_df[metric_col].sum()
    
    p_treat = float(treat_success / n_treat) if n_treat > 0 else 0.0
    p_ctrl = float(ctrl_success / n_ctrl) if n_ctrl > 0 else 0.0
    
    abs_uplift = p_treat - p_ctrl
    rel_uplift = (abs_uplift / p_ctrl) if p_ctrl > 0 else 0.0
    
    contingency = np.array([
        [treat_success, n_treat - treat_success],
        [ctrl_success, n_ctrl - ctrl_success]
    ])
    
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    except Exception as e:
        logger.warning(f"Chi-squared test failed: {e}")
        p_value = 1.0
        
    is_sig = p_value < alpha
    
    se = np.sqrt((p_treat * (1 - p_treat) / n_treat) + (p_ctrl * (1 - p_ctrl) / n_ctrl)) if (n_treat > 0 and n_ctrl > 0) else 0.0
    z = norm.ppf(1 - alpha / 2)
    ci = (abs_uplift - z * se, abs_uplift + z * se)
    
    return UpliftResult(
        treatment_rate=p_treat,
        control_rate=p_ctrl,
        absolute_uplift=abs_uplift,
        relative_uplift=rel_uplift,
        p_value=float(p_value),
        confidence_interval=ci,
        is_significant=bool(is_sig),
        sample_sizes=(n_treat, n_ctrl)
    )


def correct_multiple_testing(p_values: List[float], method: str = 'bonferroni') -> List[float]:
    """Applies Bonferroni or BH-FDR correction."""
    if method == 'bonferroni':
        m = len(p_values)
        return [min(1.0, p * m) for p in p_values]
    elif method == 'bh':
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        adjusted = np.zeros(m)
        min_adj = 1.0
        for i in range(m - 1, -1, -1):
            idx = sorted_indices[i]
            rank = i + 1
            adj_p = p_values[idx] * m / rank
            min_adj = min(min_adj, adj_p)
            adjusted[idx] = min_adj
        return adjusted.tolist()
    else:
        raise ValueError(f"Unknown correction method: {method}")


def generate_experiment_report(results_dict: Dict[str, UpliftResult], experiment_name: str, run_date: str, output_dir: Path) -> Path:
    """Saves experiment results as JSON."""
    report = {
        "experiment_name": experiment_name,
        "run_date": run_date,
        "results": {}
    }
    for k, v in results_dict.items():
        report["results"][k] = {
            "treatment_rate": v.treatment_rate,
            "control_rate": v.control_rate,
            "absolute_uplift": v.absolute_uplift,
            "relative_uplift": v.relative_uplift,
            "p_value": v.p_value,
            "confidence_interval": v.confidence_interval,
            "is_significant": v.is_significant,
            "sample_sizes": v.sample_sizes
        }
        
    out_path = output_dir / f"{experiment_name}_report_{run_date}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved experiment report to {out_path}")
    return out_path
