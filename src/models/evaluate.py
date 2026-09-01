import logging
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, List
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from src.config import get_pipeline_paths

logger = logging.getLogger(__name__)

def evaluate_churn(model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Evaluates churn binary classifier."""
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred_bin, zero_division=0)
    recall = recall_score(y_test, y_pred_bin, zero_division=0)
    f1 = f1_score(y_test, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_bin).tolist()
    
    return {
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm
    }

def evaluate_ltv(model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Evaluates LTV regressor."""
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.clip(y_pred, 0, None)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    return {
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape)
    }

def extract_feature_importance(model: lgb.Booster, feature_names: List[str]) -> pd.DataFrame:
    """Extracts and sorts feature importances from LightGBM model."""
    importance = model.feature_importance(importance_type='gain')
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    return df

def compare_models(old_metrics: Dict[str, Any], new_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Compares new model metrics with old model to recommend deployment."""
    recommendation = "rollback"
    primary_metric = ""
    improvement = 0.0
    
    if 'auc' in new_metrics and 'auc' in old_metrics:
        primary_metric = 'auc'
        improvement = new_metrics['auc'] - old_metrics['auc']
        if improvement >= 0:
            recommendation = "deploy"
    elif 'r2' in new_metrics and 'r2' in old_metrics:
        primary_metric = 'r2'
        improvement = new_metrics['r2'] - old_metrics['r2']
        if improvement >= 0:
            recommendation = "deploy"
    else:
        recommendation = "deploy"

    return {
        'recommendation': recommendation,
        'primary_metric': primary_metric,
        'improvement': float(improvement)
    }

def generate_evaluation_report(metrics: Dict[str, Any], model_name: str, run_date: str):
    """Saves evaluation report to log_dir as JSON."""
    paths = get_pipeline_paths()
    log_dir = paths['log_dir']
    
    report_path = log_dir / f"{model_name}_eval_report_{run_date}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Evaluation report saved to {report_path}")
