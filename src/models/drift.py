import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DriftReport:
    feature_name: str
    psi_value: float
    status: str
    timestamp: str

def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Calculates Population Stability Index (PSI)."""
    expected = expected.dropna()
    actual = actual.dropna()
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0
        
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)

def check_data_drift(
    reference_df: pd.DataFrame, 
    current_df: pd.DataFrame, 
    features: List[str], 
    cfg: Dict[str, Any]
) -> List[DriftReport]:
    """Checks data drift using PSI."""
    logger.info("Checking data drift...")
    psi_warning = cfg.get('psi_warning', 0.10)
    psi_critical = cfg.get('psi_critical', 0.20)
    
    reports = []
    timestamp = datetime.now().isoformat()
    
    for feature in features:
        if feature in reference_df and feature in current_df:
            psi = calculate_psi(reference_df[feature], current_df[feature])
            
            if psi >= psi_critical:
                status = "critical"
            elif psi >= psi_warning:
                status = "warning"
            else:
                status = "stable"
                
            reports.append(DriftReport(feature, psi, status, timestamp))
            
            if status != "stable":
                logger.warning(f"Drift detected for {feature}: PSI={psi:.4f} ({status})")
    
    return reports

def check_prediction_drift(old_predictions: pd.Series, new_predictions: pd.Series) -> float:
    """Checks prediction drift using PSI."""
    logger.info("Checking prediction drift...")
    psi = calculate_psi(old_predictions, new_predictions)
    return psi

def should_retrain(drift_reports: List[DriftReport], cfg: Dict[str, Any]) -> bool:
    """Determines if retraining is needed based on drift reports."""
    for report in drift_reports:
        if report.status == "critical":
            logger.warning(f"Retraining triggered due to critical drift in {report.feature_name}.")
            return True
    return False
