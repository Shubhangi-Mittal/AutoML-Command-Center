import pickle
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


class ModelServer:
    """Singleton model server that loads and serves predictions from a trained model."""

    _instance: Optional["ModelServer"] = None

    def __init__(self):
        self._active_model = None
        self._feature_names: List[str] = []
        self._model_path: Optional[str] = None
        self._job_info: Optional[Dict[str, Any]] = None
        self._deployed_at: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "ModelServer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def deploy(self, model_path: str, job_info: Dict[str, Any]) -> Dict[str, Any]:
        """Load a model from disk and make it active for serving."""
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        # Support both bundled (dict with model+feature_names) and raw model files
        if isinstance(data, dict) and "model" in data:
            self._active_model = data["model"]
            self._feature_names = data.get("feature_names", [])
        else:
            self._active_model = data
            self._feature_names = []

        self._model_path = model_path
        self._job_info = job_info
        self._deployed_at = datetime.now(timezone.utc).isoformat()

        return {
            "status": "deployed",
            "model_type": job_info.get("model_type"),
            "job_id": job_info.get("job_id"),
            "deployed_at": self._deployed_at,
        }

    def undeploy(self) -> Dict[str, str]:
        """Remove the active model."""
        self._active_model = None
        self._feature_names = []
        self._model_path = None
        self._job_info = None
        self._deployed_at = None
        return {"status": "undeployed"}

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align input DataFrame columns to match the training feature names."""
        if not self._feature_names:
            return df
        # Reindex to match training columns; missing columns become 0
        return df.reindex(columns=self._feature_names, fill_value=0)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single prediction."""
        if self._active_model is None:
            raise RuntimeError("No model deployed")

        df = self._align_features(pd.DataFrame([features]))
        prediction = self._active_model.predict(df)
        result: Dict[str, Any] = {
            "prediction": prediction.tolist(),
            "model_type": self._job_info.get("model_type") if self._job_info else None,
        }

        if hasattr(self._active_model, "predict_proba"):
            probabilities = self._active_model.predict_proba(df)
            result["probabilities"] = probabilities.tolist()

        return result

    def predict_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch predictions."""
        if self._active_model is None:
            raise RuntimeError("No model deployed")

        df = self._align_features(pd.DataFrame(records))
        predictions = self._active_model.predict(df)
        result: Dict[str, Any] = {"predictions": predictions.tolist()}

        if hasattr(self._active_model, "predict_proba"):
            probabilities = self._active_model.predict_proba(df)
            result["probabilities"] = probabilities.tolist()

        return result

    def get_status(self) -> Dict[str, Any]:
        """Return current serving status."""
        if self._active_model is None:
            return {"status": "no_model", "model_type": None, "job_id": None}

        return {
            "status": "deployed",
            "model_type": self._job_info.get("model_type") if self._job_info else None,
            "job_id": self._job_info.get("job_id") if self._job_info else None,
            "metrics": self._job_info.get("metrics") if self._job_info else None,
            "deployed_at": self._deployed_at,
        }
