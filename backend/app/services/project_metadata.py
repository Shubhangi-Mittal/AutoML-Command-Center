import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.config import settings


class ProjectMetadataStore:
    """JSON-backed metadata store for lightweight portfolio features."""

    def __init__(self):
        self.base_dir = settings.APP_DATA_DIR
        os.makedirs(self.base_dir, exist_ok=True)
        self.datasets_path = os.path.join(self.base_dir, "dataset_versions.json")
        self.experiments_path = os.path.join(self.base_dir, "experiment_metadata.json")
        self.predictions_path = os.path.join(self.base_dir, "prediction_history.json")

    def register_dataset(self, dataset_id: str, dataset_name: str, file_path: str) -> Dict[str, Any]:
        payload = self._load(self.datasets_path, {"datasets": {}})
        datasets = payload["datasets"]

        family_key = dataset_name.strip().lower()
        previous_versions = [
            info for info in datasets.values()
            if info.get("family_key") == family_key
        ]
        version = max((info.get("version", 0) for info in previous_versions), default=0) + 1
        previous_dataset_id = None
        if previous_versions:
            previous_dataset_id = sorted(
                previous_versions,
                key=lambda item: item.get("version", 0),
                reverse=True,
            )[0].get("dataset_id")

        metadata = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "family_key": family_key,
            "version": version,
            "previous_dataset_id": previous_dataset_id,
            "file_hash": self._file_hash(file_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        datasets[dataset_id] = metadata
        self._save(self.datasets_path, payload)
        return metadata

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        payload = self._load(self.datasets_path, {"datasets": {}})
        return payload["datasets"].get(dataset_id, {})

    def list_dataset_versions(self, dataset_id: str) -> List[Dict[str, Any]]:
        payload = self._load(self.datasets_path, {"datasets": {}})
        current = payload["datasets"].get(dataset_id)
        if not current:
            return []
        family_key = current.get("family_key")
        versions = [
            info for info in payload["datasets"].values()
            if info.get("family_key") == family_key
        ]
        return sorted(versions, key=lambda item: item.get("version", 0))

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        payload = self._load(self.experiments_path, {"experiments": {}})
        return payload["experiments"].get(experiment_id, {
            "tags": [],
            "favorite": False,
            "archived": False,
            "notes": "",
        })

    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._load(self.experiments_path, {"experiments": {}})
        experiments = payload["experiments"]
        current = experiments.get(experiment_id, {
            "tags": [],
            "favorite": False,
            "archived": False,
            "notes": "",
        })
        current.update({k: v for k, v in updates.items() if v is not None})
        experiments[experiment_id] = current
        self._save(self.experiments_path, payload)
        return current

    def log_prediction(self, entry: Dict[str, Any]) -> None:
        payload = self._load(self.predictions_path, {"predictions": []})
        predictions = payload["predictions"]
        predictions.append(entry)
        payload["predictions"] = predictions[-100:]
        self._save(self.predictions_path, payload)

    def list_predictions(self, dataset_id: str | None = None) -> List[Dict[str, Any]]:
        payload = self._load(self.predictions_path, {"predictions": []})
        predictions = payload["predictions"]
        if dataset_id:
            predictions = [item for item in predictions if item.get("dataset_id") == dataset_id]
        return list(reversed(predictions))

    def _load(self, path: str, default: Dict[str, Any]) -> Dict[str, Any]:
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return default

    def _save(self, path: str, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def _file_hash(self, path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()


metadata_store = ProjectMetadataStore()
