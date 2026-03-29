from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.job import TrainingJob
from app.services.serving import ModelServer


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

    # Find the model path from the models directory
    import os, glob
    from app.config import settings

    model_type = job.model_type
    pattern = os.path.join(settings.MODEL_DIR, f"{model_type}_*.pkl")
    model_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    if not model_files:
        raise HTTPException(status_code=404, detail=f"No model file found for {model_type}")

    model_path = model_files[0]
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
def serving_status():
    server = ModelServer.get_instance()
    return server.get_status()
