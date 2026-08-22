"""
Scoring module for churn and LTV.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def score_churn(df: pd.DataFrame, model, feature_columns: list[str]) -> pd.DataFrame:
    """Adds churn_probability column."""
    logger.info("Scoring churn...")
    df_out = df.copy()
    X = df_out[feature_columns]
    
    try:
        best_iteration = getattr(model, 'best_iteration', None)
        if best_iteration is not None:
            df_out['churn_probability'] = model.predict(X, num_iteration=best_iteration)
        else:
            df_out['churn_probability'] = model.predict(X)
    except Exception as e:
        logger.error(f"Failed to score churn: {e}")
        df_out['churn_probability'] = np.nan
        
    return df_out

def score_ltv(df: pd.DataFrame, model, feature_columns: list[str]) -> pd.DataFrame:
    """Adds predicted_ltv column."""
    logger.info("Scoring LTV...")
    df_out = df.copy()
    X = df_out[feature_columns]
    
    try:
        best_iteration = getattr(model, 'best_iteration', None)
        if best_iteration is not None:
            preds = model.predict(X, num_iteration=best_iteration)
        else:
            preds = model.predict(X)
        df_out['predicted_ltv'] = np.clip(preds, 0, None)
    except Exception as e:
        logger.error(f"Failed to score LTV: {e}")
        df_out['predicted_ltv'] = np.nan
        
    return df_out

def classify_risk(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Adds risk_level column (High/Medium/Low)."""
    logger.info("Classifying risk...")
    df_out = df.copy()
    
    if 'churn_probability' not in df_out.columns:
        raise ValueError("churn_probability column is missing. Run score_churn first.")
        
    def get_risk(prob):
        if pd.isna(prob): return 'Unknown'
        elif prob >= threshold: return 'High'
        elif prob >= threshold * 0.5: return 'Medium'
        else: return 'Low'
        
    df_out['risk_level'] = df_out['churn_probability'].apply(get_risk)
    return df_out
