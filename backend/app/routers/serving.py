from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.dataset import Dataset
from app.models.job import TrainingJob
from app.services.serving import ModelServer
from app.services.trainer import resolve_model_artifact


router = APIRouter()


class DeployRequest(BaseModel):
    job_id: str


class PredictionRequest(BaseModel):
    features: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    records: List[Dict[str, Any]]


@router.post("/deploy")
def deploy_model(request: DeployRequest, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(TrainingJob.id == request.job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Can only deploy completed jobs")

    model_path = resolve_model_artifact(job.model_type, job.id)
    if not model_path:
        raise HTTPException(status_code=404, detail=f"No model file found for {job.model_type}")

    job_info = {
        "job_id": job.id,
        "model_type": job.model_type,
        "metrics": job.metrics,
        "dataset_id": job.dataset_id,
    }

    server = ModelServer.get_instance()
    result = server.deploy(model_path, job_info)
    return result


@router.post("/undeploy")
def undeploy_model():
    server = ModelServer.get_instance()
    return server.undeploy()


@router.post("/predict")
def predict(request: PredictionRequest):
    server = ModelServer.get_instance()
    try:
        return server.predict(request.features)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    server = ModelServer.get_instance()
    try:
        return server.predict_batch(request.records)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status")
def serving_status(db: Session = Depends(get_db)):
    server = ModelServer.get_instance()
    status = server.get_status()

    if status.get("status") == "deployed" and status.get("job_id"):
        job = db.query(TrainingJob).filter(TrainingJob.id == status["job_id"]).first()
        if job:
            dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
            status["dataset_id"] = job.dataset_id
            if dataset:
                status["dataset_name"] = dataset.name
                status["target_column"] = dataset.target_column

    return status


@router.get("/template/{dataset_id}")
def get_prediction_template(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    import pandas as pd

    df = pd.read_csv(dataset.file_path, encoding="utf-8-sig", nrows=1)
    feature_cols = [col for col in df.columns if col != dataset.target_column]
    sample_row = df[feature_cols].iloc[0].to_dict() if len(df.index) else {}

    for key, value in sample_row.items():
        if pd.isna(value):
            sample_row[key] = None

    return {
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "target_column": dataset.target_column,
        "task_type": dataset.task_type,
        "feature_columns": feature_cols,
        "sample_input": sample_row,
    }
