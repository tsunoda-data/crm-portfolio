import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import lightgbm as lgb

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_model_path(self, model_name: str, version: int, run_date: str) -> Path:
        return self.model_dir / f"{model_name}_v{version}_{run_date}.pkl"

    def _get_meta_path(self, model_name: str, version: int, run_date: str) -> Path:
        return self.model_dir / f"{model_name}_v{version}_{run_date}.json"

    def get_latest_version(self, model_name: str) -> int:
        """Returns the latest version number for a model."""
        versions = self.list_versions(model_name)
        if not versions:
            return 0
        return max(v['version'] for v in versions)

    def save_model(self, model: lgb.Booster, model_name: str, version: int, metrics: Dict[str, Any], feature_columns: List[str], run_date: str):
        """Saves a model and its metadata."""
        model_path = self._get_model_path(model_name, version, run_date)
        meta_path = self._get_meta_path(model_name, version, run_date)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        metadata = {
            'model_name': model_name,
            'version': version,
            'date': run_date,
            'metrics': metrics,
            'feature_columns': feature_columns,
            'status': 'active',
            'model_file': model_path.name
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Saved {model_name} v{version} to {model_path}")

    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Lists all versions of a model."""
        versions = []
        for meta_file in self.model_dir.glob(f"{model_name}_v*.json"):
            with open(meta_file, 'r', encoding='utf-8') as f:
                try:
                    meta = json.load(f)
                    versions.append(meta)
                except json.JSONDecodeError:
                    continue
        return sorted(versions, key=lambda x: x['version'])

    def _find_meta_by_version(self, model_name: str, version: int) -> Optional[Path]:
        for meta_file in self.model_dir.glob(f"{model_name}_v{version}_*.json"):
            return meta_file
        return None

    def load_model(self, model_name: str, version: str | int = 'latest') -> tuple[lgb.Booster, Dict[str, Any]]:
        """Loads a model and its metadata."""
        if version == 'latest':
            version = self.get_latest_version(model_name)
            
        if version == 0:
            raise FileNotFoundError(f"No versions found for model {model_name}")
            
        meta_file = self._find_meta_by_version(model_name, int(version))
        if not meta_file:
            raise FileNotFoundError(f"Model {model_name} v{version} not found")
            
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        model_path = self.model_dir / metadata['model_file']
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        return model, metadata

    def rollback(self, model_name: str, to_version: int):
        """Sets a specific version as active and others as retired."""
        versions = self.list_versions(model_name)
        for v in versions:
            v_num = v['version']
            status = 'active' if v_num == to_version else 'retired'
            if v.get('status') != status:
                v['status'] = status
                meta_file = self._find_meta_by_version(model_name, v_num)
                if meta_file:
                    with open(meta_file, 'w', encoding='utf-8') as f:
                        json.dump(v, f, indent=4, ensure_ascii=False)
        logger.info(f"Rolled back {model_name} to version {to_version}")

    def compare_versions(self, model_name: str, v1: int, v2: int) -> Dict[str, Any]:
        """Compares metrics between two versions."""
        _, meta1 = self.load_model(model_name, v1)
        _, meta2 = self.load_model(model_name, v2)
        
        return {
            'v1': meta1['metrics'],
            'v2': meta2['metrics']
        }
