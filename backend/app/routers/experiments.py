from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.experiment import Experiment
from app.models.job import TrainingJob
from app.services.experiment_tracker import ExperimentTracker
from app.services.project_metadata import metadata_store


router = APIRouter()


class CreateExperimentRequest(BaseModel):
    dataset_id: str
    name: str
    optimization_metric: Optional[str] = "f1"
    description: Optional[str] = ""


class UpdateExperimentMetadataRequest(BaseModel):
    name: Optional[str] = None
    tags: Optional[list[str]] = None
    favorite: Optional[bool] = None
    archived: Optional[bool] = None
    notes: Optional[str] = None


@router.post("/")
def create_experiment(request: CreateExperimentRequest, db: Session = Depends(get_db)):
    tracker = ExperimentTracker(db)
    experiment = tracker.create_experiment(
        dataset_id=request.dataset_id,
        name=request.name,
        optimization_metric=request.optimization_metric or "f1",
        description=request.description or "",
    )
    return {
        "id": experiment.id,
        "dataset_id": experiment.dataset_id,
        "name": experiment.name,
        "optimization_metric": experiment.optimization_metric,
        "status": experiment.status,
        "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
    }


@router.get("/")
def list_experiments(dataset_id: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Experiment)
    if dataset_id:
        query = query.filter(Experiment.dataset_id == dataset_id)
    experiments = query.order_by(Experiment.created_at.desc()).all()
    return [
        _merge_experiment_metadata({
            "id": e.id,
            "dataset_id": e.dataset_id,
            "name": e.name,
            "optimization_metric": e.optimization_metric,
            "best_job_id": e.best_job_id,
            "status": e.status,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }, e.id)
        for e in experiments
    ]


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    jobs = (
        db.query(TrainingJob)
        .filter(TrainingJob.experiment_id == experiment_id)
        .order_by(TrainingJob.created_at.desc())
        .all()
    )

    return _merge_experiment_metadata({
        "id": experiment.id,
        "dataset_id": experiment.dataset_id,
        "name": experiment.name,
        "description": experiment.description,
        "optimization_metric": experiment.optimization_metric,
        "best_job_id": experiment.best_job_id,
        "status": experiment.status,
        "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
        "jobs": [
            {
                "id": j.id,
                "model_type": j.model_type,
                "status": j.status,
                "metrics": j.metrics,
                "feature_importance": j.feature_importance,
                "hyperparameters": j.hyperparameters,
                "training_duration_seconds": j.training_duration_seconds,
            }
            for j in jobs
        ],
    }, experiment.id)


@router.get("/{experiment_id}/compare")
def compare_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Return all completed jobs sorted by optimization metric with best model indicator."""
    tracker = ExperimentTracker(db)
    try:
        return tracker.compare_jobs(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{experiment_id}/complete")
def complete_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Mark an experiment as completed and determine the best model."""
    tracker = ExperimentTracker(db)
    try:
        experiment = tracker.complete_experiment(experiment_id)
        return {
            "id": experiment.id,
            "status": experiment.status,
            "best_job_id": experiment.best_job_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/{experiment_id}/metadata")
def update_experiment_metadata(
    experiment_id: str,
    request: UpdateExperimentMetadataRequest,
    db: Session = Depends(get_db),
):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if request.name:
        experiment.name = request.name
        db.commit()

    metadata = metadata_store.update_experiment(experiment_id, request.model_dump(exclude_none=True))
    return {
        "id": experiment_id,
        "name": experiment.name,
        **metadata,
    }


@router.get("/{experiment_id}/report")
def experiment_report(experiment_id: str, db: Session = Depends(get_db)):
    tracker = ExperimentTracker(db)
    try:
        comparison = tracker.compare_jobs(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    report_lines = [
        f"# Experiment Report: {comparison.get('name', 'Experiment')}",
        "",
        f"- Experiment ID: `{comparison['experiment_id']}`",
        f"- Dataset ID: `{comparison['dataset_id']}`",
        f"- Optimization metric: `{comparison['optimization_metric']}`",
        f"- Status: `{comparison['status']}`",
        "",
        "## Model Results",
        "",
    ]

    for job in comparison["jobs"]:
        metrics = ", ".join(
            f"{key}={value}"
            for key, value in (job.get("metrics") or {}).items()
            if key != "confusion_matrix"
        )
        report_lines.append(
            f"- {'BEST ' if job.get('is_best') else ''}{job['model_type']}: {metrics}"
        )

    return {"markdown": "\n".join(report_lines)}


def _merge_experiment_metadata(payload: dict, experiment_id: str):
    payload.update(metadata_store.get_experiment(experiment_id))
    return payload
