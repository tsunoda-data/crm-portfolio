import logging
import lightgbm as lgb
import pandas as pd
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def temporal_train_test_split(
    df: pd.DataFrame, 
    date_column: str, 
    split_date: str | pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data temporally. Train on data before split_date, test on data on/after split_date.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    split_date = pd.to_datetime(split_date)
    
    train_df = df[df[date_column] < split_date].copy()
    test_df = df[df[date_column] >= split_date].copy()
    
    logger.info(f"Temporal split at {split_date}. Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df

def train_churn_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series, 
    cfg: Dict[str, Any]
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """
    Trains LightGBM binary classifier for churn prediction.
    """
    logger.info("Training churn model (LightGBM binary classifier)...")
    
    train_dataset = lgb.Dataset(X_train, label=y_train)
    valid_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
    
    params = {
        'objective': cfg.get('objective', 'binary'),
        'metric': cfg.get('metric', 'auc'),
        'num_leaves': cfg.get('num_leaves', 63),
        'learning_rate': cfg.get('learning_rate', 0.05),
        'is_unbalance': True,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    evals_result = {}
    model = lgb.train(
        params,
        train_set=train_dataset,
        num_boost_round=cfg.get('num_boost_round', 500),
        valid_sets=[train_dataset, valid_dataset],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(cfg.get('early_stopping_rounds', 50), verbose=False),
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals_result)
        ]
    )
    
    metrics = {
        'best_iteration': model.best_iteration,
        'best_score': model.best_score['valid'][cfg.get('metric', 'auc')],
        'evals_result': evals_result
    }
    
    logger.info(f"Churn model training completed. Best AUC: {metrics['best_score']:.4f}")
    return model, metrics

def train_ltv_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series, 
    cfg: Dict[str, Any]
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """
    Trains LightGBM regressor for LTV prediction.
    """
    logger.info("Training LTV model (LightGBM regressor)...")
    
    train_dataset = lgb.Dataset(X_train, label=y_train)
    valid_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
    
    params = {
        'objective': cfg.get('objective', 'regression'),
        'metric': cfg.get('metric', 'rmse'),
        'num_leaves': cfg.get('num_leaves', 127),
        'learning_rate': cfg.get('learning_rate', 0.05),
        'lambda_l2': 1.0,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    evals_result = {}
    model = lgb.train(
        params,
        train_set=train_dataset,
        num_boost_round=cfg.get('num_boost_round', 500),
        valid_sets=[train_dataset, valid_dataset],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(cfg.get('early_stopping_rounds', 50), verbose=False),
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals_result)
        ]
    )
    
    metrics = {
        'best_iteration': model.best_iteration,
        'best_score': model.best_score['valid'][cfg.get('metric', 'rmse')],
        'evals_result': evals_result
    }
    
    logger.info(f"LTV model training completed. Best RMSE: {metrics['best_score']:.4f}")
    return model, metrics
