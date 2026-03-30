import pickle
import pandas as pd
import time
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
        self._prediction_count = 0
        self._avg_latency_ms = 0.0
        self._last_prediction_at: Optional[str] = None

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
        self._prediction_count = 0
        self._avg_latency_ms = 0.0
        self._last_prediction_at = None

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
        self._prediction_count = 0
        self._avg_latency_ms = 0.0
        self._last_prediction_at = None
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

        start_time = time.perf_counter()
        df = self._align_features(pd.DataFrame([features]))
        prediction = self._active_model.predict(df)
        result: Dict[str, Any] = {
            "prediction": prediction.tolist(),
            "model_type": self._job_info.get("model_type") if self._job_info else None,
        }

        if hasattr(self._active_model, "predict_proba"):
            probabilities = self._active_model.predict_proba(df)
            result["probabilities"] = probabilities.tolist()

        self._record_latency(time.perf_counter() - start_time)
        return result

    def predict_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch predictions."""
        if self._active_model is None:
            raise RuntimeError("No model deployed")

        start_time = time.perf_counter()
        df = self._align_features(pd.DataFrame(records))
        predictions = self._active_model.predict(df)
        result: Dict[str, Any] = {"predictions": predictions.tolist()}

        if hasattr(self._active_model, "predict_proba"):
            probabilities = self._active_model.predict_proba(df)
            result["probabilities"] = probabilities.tolist()

        self._record_latency(time.perf_counter() - start_time, batch_size=len(records))
        return result

    def explain_prediction(self, features: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """Generate lightweight feature-impact explanations for a prediction."""
        if self._active_model is None:
            raise RuntimeError("No model deployed")

        df = self._align_features(pd.DataFrame([features]))
        base_result = self.predict(features)
        impacts = []

        for column in df.columns:
            ablated = df.copy()
            ablated.loc[0, column] = 0
            alternative_pred = self._active_model.predict(ablated)
            score = self._prediction_delta(base_result, alternative_pred.tolist(), ablated)
            impacts.append({
                "feature": column,
                "impact": round(float(score), 6),
                "value": df.loc[0, column].item() if hasattr(df.loc[0, column], "item") else df.loc[0, column],
            })

        impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return {
            "model_type": self._job_info.get("model_type") if self._job_info else None,
            "prediction": base_result.get("prediction"),
            "top_contributors": impacts[:top_k],
        }

    def get_status(self) -> Dict[str, Any]:
        """Return current serving status."""
        if self._active_model is None:
            return {"status": "no_model", "model_type": None, "job_id": None}

        return {
            "status": "deployed",
            "dataset_id": self._job_info.get("dataset_id") if self._job_info else None,
            "model_type": self._job_info.get("model_type") if self._job_info else None,
            "job_id": self._job_info.get("job_id") if self._job_info else None,
            "metrics": self._job_info.get("metrics") if self._job_info else None,
            "deployed_at": self._deployed_at,
            "prediction_count": self._prediction_count,
            "avg_latency_ms": round(self._avg_latency_ms, 3),
            "last_prediction_at": self._last_prediction_at,
        }

    def _record_latency(self, duration_seconds: float, batch_size: int = 1) -> None:
        latency_ms = duration_seconds * 1000
        total_predictions = self._prediction_count + batch_size
        if total_predictions <= 0:
            return
        self._avg_latency_ms = (
            (self._avg_latency_ms * self._prediction_count) + (latency_ms * batch_size)
        ) / total_predictions
        self._prediction_count = total_predictions
        self._last_prediction_at = datetime.now(timezone.utc).isoformat()

    def _prediction_delta(self, base_result: Dict[str, Any], alt_prediction: List[Any], alt_df: pd.DataFrame) -> float:
        if hasattr(self._active_model, "predict_proba"):
            base_proba = self._active_model.predict_proba(self._align_features(pd.DataFrame([alt_df.iloc[0].to_dict()])))
            current_proba = base_result.get("probabilities")
            if current_proba:
                return max(abs(current_proba[0][idx] - base_proba[0][idx]) for idx in range(len(base_proba[0])))
        base_prediction = base_result.get("prediction", [0])
        return abs(float(base_prediction[0]) - float(alt_prediction[0]))
