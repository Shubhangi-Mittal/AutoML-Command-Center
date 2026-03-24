import pandas as pd
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.dataset import Dataset
from app.models.job import TrainingJob
from app.services.feature_engine import FeatureEngine
from app.services.trainer import train_model, train_all_models


router = APIRouter()


class TrainRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    model_types: Optional[List[str]] = None
    experiment_id: Optional[str] = None


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

    df = pd.read_csv(dataset.file_path, encoding="utf-8-sig")

    engine = FeatureEngine(df, target_col, task_type)
    X_train, X_test, y_train, y_test, metadata = engine.auto_engineer()

    model_types = request.model_types or ["linear", "xgboost", "random_forest"]
    results = train_all_models(X_train, X_test, y_train, y_test, task_type, model_types)

    jobs = []
    for result in results:
        job = TrainingJob(
            dataset_id=dataset.id,
            experiment_id=request.experiment_id,
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

    return {
        "dataset_id": dataset.id,
        "task_type": task_type,
        "target_column": target_col,
        "feature_engineering": metadata,
        "jobs": jobs,
    }


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
