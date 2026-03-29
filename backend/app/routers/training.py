import json
import asyncio
import pandas as pd
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.dataset import Dataset
from app.models.job import TrainingJob
from app.services.feature_engine import FeatureEngine
from app.services.trainer import train_model, train_all_models
from app.services.experiment_tracker import ExperimentTracker


router = APIRouter()


class TrainRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    model_types: Optional[List[str]] = None
    experiment_id: Optional[str] = None
    async_mode: Optional[bool] = False


@router.post("/launch")
def launch_training(request: TrainRequest, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    target_col = request.target_column or dataset.target_column
    task_type = request.task_type or dataset.task_type
    if not target_col or not task_type:
        raise HTTPException(
            status_code=400,
            detail="target_column and task_type are required",
        )

    model_types = request.model_types or ["linear", "xgboost", "random_forest"]

    # Auto-create experiment if none provided
    experiment_id = request.experiment_id
    if not experiment_id:
        tracker = ExperimentTracker(db)
        experiment = tracker.create_experiment(
            dataset_id=dataset.id,
            name=f"{dataset.name} - {', '.join(model_types)}",
            optimization_metric="f1" if task_type == "classification" else "rmse",
        )
        experiment_id = experiment.id

    # Async mode: dispatch to Celery
    if request.async_mode:
        return _launch_async(dataset, target_col, task_type, model_types, experiment_id, db)

    # Sync mode: train immediately (default for simplicity)
    return _launch_sync(dataset, target_col, task_type, model_types, experiment_id, db)


def _launch_async(dataset, target_col, task_type, model_types, experiment_id, db):
    """Create pending jobs and dispatch Celery tasks."""
    from app.tasks.training_tasks import train_single_model_task

    jobs = []
    for model_type in model_types:
        job = TrainingJob(
            dataset_id=dataset.id,
            experiment_id=experiment_id,
            status="pending",
            model_type=model_type,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # Dispatch Celery task
        task = train_single_model_task.delay(
            job_id=job.id,
            dataset_id=dataset.id,
            file_path=dataset.file_path,
            target_col=target_col,
            task_type=task_type,
            model_type=model_type,
        )

        jobs.append({
            "job_id": job.id,
            "model_type": model_type,
            "status": "pending",
            "celery_task_id": task.id,
        })

    return {
        "dataset_id": dataset.id,
        "experiment_id": experiment_id,
        "task_type": task_type,
        "target_column": target_col,
        "async": True,
        "jobs": jobs,
    }


def _launch_sync(dataset, target_col, task_type, model_types, experiment_id, db):
    """Train all models synchronously and return results."""
    df = pd.read_csv(dataset.file_path, encoding="utf-8-sig")

    engine = FeatureEngine(df, target_col, task_type)
    X_train, X_test, y_train, y_test, metadata = engine.auto_engineer()

    results = train_all_models(X_train, X_test, y_train, y_test, task_type, model_types)

    jobs = []
    for result in results:
        job = TrainingJob(
            dataset_id=dataset.id,
            experiment_id=experiment_id,
            status="completed",
            model_type=result["model_type"],
            hyperparameters=result.get("hyperparameters"),
            metrics=result["metrics"],
            feature_importance=result.get("feature_importance"),
            training_duration_seconds=result["training_duration_seconds"],
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        jobs.append({
            "job_id": job.id,
            "model_type": result["model_type"],
            "metrics": result["metrics"],
            "feature_importance": result.get("feature_importance"),
            "training_duration_seconds": result["training_duration_seconds"],
            "model_path": result["model_path"],
        })

    # Auto-complete experiment
    if experiment_id:
        tracker = ExperimentTracker(db)
        try:
            tracker.complete_experiment(experiment_id)
        except ValueError:
            pass

    return {
        "dataset_id": dataset.id,
        "experiment_id": experiment_id,
        "task_type": task_type,
        "target_column": target_col,
        "async": False,
        "feature_engineering": metadata,
        "jobs": jobs,
    }


@router.get("/stream/progress/{dataset_id}")
async def stream_progress(dataset_id: str):
    """SSE endpoint streaming real-time training progress via Redis pub/sub."""
    import redis as redis_lib

    async def event_generator():
        try:
            r = redis_lib.from_url(settings.REDIS_URL)
            pubsub = r.pubsub()
            pubsub.subscribe("training_progress")

            while True:
                message = pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    if data.get("dataset_id") == dataset_id:
                        yield f"data: {json.dumps(data)}\n\n"

                        # Stop streaming when all jobs are done
                        if data.get("status") in ("completed", "failed"):
                            yield f"data: {json.dumps({'status': 'stream_end'})}\n\n"

                # Heartbeat to keep connection alive
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.5)

        except Exception:
            yield f"data: {json.dumps({'status': 'error', 'detail': 'Stream disconnected'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/")
def list_training_jobs(dataset_id: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(TrainingJob)
    if dataset_id:
        query = query.filter(TrainingJob.dataset_id == dataset_id)
    jobs = query.order_by(TrainingJob.created_at.desc()).all()
    return [
        {
            "id": j.id,
            "dataset_id": j.dataset_id,
            "experiment_id": j.experiment_id,
            "status": j.status,
            "model_type": j.model_type,
            "metrics": j.metrics,
            "feature_importance": j.feature_importance,
            "training_duration_seconds": j.training_duration_seconds,
            "created_at": j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]


@router.get("/{job_id}")
def get_training_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return {
        "id": job.id,
        "dataset_id": job.dataset_id,
        "experiment_id": job.experiment_id,
        "status": job.status,
        "model_type": job.model_type,
        "hyperparameters": job.hyperparameters,
        "metrics": job.metrics,
        "feature_importance": job.feature_importance,
        "training_duration_seconds": job.training_duration_seconds,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
