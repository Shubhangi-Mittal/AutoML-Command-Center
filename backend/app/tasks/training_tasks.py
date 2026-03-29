import json
import pandas as pd
import redis
from datetime import datetime, timezone

from celery_app import celery_app
from app.config import settings
from app.database import SessionLocal
from app.models.dataset import Dataset  # noqa: F401
from app.models.experiment import Experiment  # noqa: F401
from app.models.job import TrainingJob
from app.services.feature_engine import FeatureEngine
from app.services.trainer import train_model


def _publish_progress(dataset_id: str, job_id: str, model_type: str, status: str, detail: str = ""):
    """Publish training progress to Redis pub/sub."""
    try:
        r = redis.from_url(settings.REDIS_URL)
        message = json.dumps({
            "dataset_id": dataset_id,
            "job_id": job_id,
            "model_type": model_type,
            "status": status,
            "detail": detail,
        })
        r.publish("training_progress", message)
    except Exception:
        pass  # Don't fail training if Redis pub/sub fails


@celery_app.task(bind=True, name="train_single_model")
def train_single_model_task(
    self,
    job_id: str,
    dataset_id: str,
    file_path: str,
    target_col: str,
    task_type: str,
    model_type: str,
    hyperparameters: dict = None,
):
    """Celery task to train a single model asynchronously."""
    db = SessionLocal()
    try:
        # Update job status to running
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            return {"error": "Job not found"}

        job.status = "running"
        job.celery_task_id = self.request.id
        db.commit()

        _publish_progress(dataset_id, job_id, model_type, "running", "Loading data...")

        # Load and engineer features
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        engine = FeatureEngine(df, target_col, task_type)
        X_train, X_test, y_train, y_test, metadata = engine.auto_engineer()

        _publish_progress(dataset_id, job_id, model_type, "running", "Training model...")

        # Train model
        result = train_model(
            X_train, X_test, y_train, y_test,
            model_type, task_type, hyperparameters or {},
        )

        _publish_progress(dataset_id, job_id, model_type, "running", "Saving results...")

        # Update job with results
        job.status = "completed"
        job.metrics = result["metrics"]
        job.feature_importance = result.get("feature_importance")
        job.hyperparameters = result.get("hyperparameters")
        job.training_duration_seconds = result["training_duration_seconds"]
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

        _publish_progress(dataset_id, job_id, model_type, "completed", json.dumps(result["metrics"]))

        return {
            "job_id": job_id,
            "model_type": model_type,
            "status": "completed",
            "metrics": result["metrics"],
            "feature_importance": result.get("feature_importance"),
            "training_duration_seconds": result["training_duration_seconds"],
            "model_path": result["model_path"],
        }

    except Exception as e:
        # Mark job as failed
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.status = "failed"
            db.commit()
        _publish_progress(dataset_id, job_id, model_type, "failed", str(e))
        raise

    finally:
        db.close()
