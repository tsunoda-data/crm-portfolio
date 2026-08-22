"""
Data quality monitoring module.
"""
import logging
import yaml
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class DataQualityReport:
    """Stores the result of data quality checks."""
    passed: bool
    checks: list[dict[str, Any]]
    timestamp: str
    summary: str

class DataQualityChecker:
    """Runs data quality checks on DataFrames."""
    
    def __init__(self, config: str | Path | dict | None = None):
        self.rules = {}
        if config is None:
            return
        # If a dict is passed (pipeline config), resolve the rules YAML path
        if isinstance(config, dict):
            config_dir = Path(__file__).resolve().parent.parent.parent / "config"
            path = config_dir / "data_quality_rules.yaml"
        else:
            path = Path(config)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.rules = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Quality rules config not found at {path}")
                
    def check_schema(self, df: pd.DataFrame, expected_columns: list[str], expected_dtypes: dict | None = None) -> list[str]:
        """Checks if all expected columns are present."""
        issues = []
        missing = set(expected_columns) - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")
        return issues
        
    def check_null_rates(self, df: pd.DataFrame, max_null_rates: dict[str, float]) -> list[str]:
        """Checks if null rates exceed maximum allowed."""
        issues = []
        for col, max_rate in max_null_rates.items():
            if col in df.columns:
                rate = df[col].isnull().mean()
                if rate > max_rate:
                    issues.append(f"Column {col} has null rate {rate:.2f} > {max_rate}")
        return issues
        
    def check_value_ranges(self, df: pd.DataFrame, ranges_dict: dict[str, dict[str, float]]) -> list[str]:
        """Checks if numeric values fall within expected ranges."""
        issues = []
        for col, limits in ranges_dict.items():
            if col in df.columns:
                if 'min' in limits and df[col].min() < limits['min']:
                    issues.append(f"Column {col} has values < {limits['min']}")
                if 'max' in limits and df[col].max() > limits['max']:
                    issues.append(f"Column {col} has values > {limits['max']}")
        return issues
        
    def check_row_count(self, df: pd.DataFrame, min_rows: int, max_rows: int) -> list[str]:
        """Checks if row count is within bounds."""
        issues = []
        count = len(df)
        if count < min_rows:
            issues.append(f"Row count {count} < min {min_rows}")
        if count > max_rows:
            issues.append(f"Row count {count} > max {max_rows}")
        return issues
        
    def check_duplicates(self, df: pd.DataFrame, key_column: str) -> list[str]:
        """Checks for duplicate primary keys."""
        issues = []
        if key_column in df.columns:
            dups = df.duplicated(subset=[key_column]).sum()
            if dups > 0:
                issues.append(f"Found {dups} duplicate rows for key {key_column}")
        return issues
        
    def run_all_checks(self, df: pd.DataFrame) -> DataQualityReport:
        """Runs all checks based on loaded rules."""
        logger.info("Running data quality checks...")
        all_issues = []
        
        if not self.rules:
            return DataQualityReport(True, [], datetime.now().isoformat(), "No rules loaded")
            
        if 'expected_columns' in self.rules:
            all_issues.extend(self.check_schema(df, self.rules['expected_columns']))
            
        if 'max_null_rates' in self.rules:
            all_issues.extend(self.check_null_rates(df, self.rules['max_null_rates']))
            
        if 'value_ranges' in self.rules:
            all_issues.extend(self.check_value_ranges(df, self.rules['value_ranges']))
            
        if 'row_count' in self.rules:
            limits = self.rules['row_count']
            all_issues.extend(self.check_row_count(df, limits.get('min', 0), limits.get('max', float('inf'))))
            
        if 'key_column' in self.rules:
            all_issues.extend(self.check_duplicates(df, self.rules['key_column']))
            
        passed = len(all_issues) == 0
        summary = "Passed all checks" if passed else f"Failed {len(all_issues)} checks"
        
        checks = [{"issue": i} for i in all_issues]
        
        return DataQualityReport(
            passed=passed,
            checks=checks,
            timestamp=datetime.now().isoformat(),
            summary=summary
        )
