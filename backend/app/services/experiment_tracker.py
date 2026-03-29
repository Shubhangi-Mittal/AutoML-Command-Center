from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.models.experiment import Experiment
from app.models.job import TrainingJob


class ExperimentTracker:
    """Lightweight experiment tracker that stores everything in PostgreSQL."""

    def __init__(self, db: Session):
        self.db = db

    def create_experiment(
        self,
        dataset_id: str,
        name: str,
        optimization_metric: str = "f1",
        description: str = "",
    ) -> Experiment:
        """Create a new experiment."""
        experiment = Experiment(
            dataset_id=dataset_id,
            name=name,
            optimization_metric=optimization_metric,
            description=description,
            status="running",
        )
        self.db.add(experiment)
        self.db.commit()
        self.db.refresh(experiment)
        return experiment

    def log_job_to_experiment(self, experiment_id: str, job_id: str) -> None:
        """Associate a training job with an experiment."""
        job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.experiment_id = experiment_id
            self.db.commit()

    def complete_experiment(self, experiment_id: str) -> Experiment:
        """Mark experiment as completed and determine the best job."""
        experiment = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        jobs = (
            self.db.query(TrainingJob)
            .filter(
                TrainingJob.experiment_id == experiment_id,
                TrainingJob.status == "completed",
            )
            .all()
        )

        if jobs:
            best_job = self._find_best_job(jobs, experiment.optimization_metric)
            experiment.best_job_id = best_job.id

        experiment.status = "completed"
        self.db.commit()
        self.db.refresh(experiment)
        return experiment

    def compare_jobs(self, experiment_id: str) -> Dict[str, Any]:
        """Return all completed jobs for an experiment, sorted by optimization metric."""
        experiment = self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        jobs = (
            self.db.query(TrainingJob)
            .filter(
                TrainingJob.experiment_id == experiment_id,
                TrainingJob.status == "completed",
            )
            .all()
        )

        metric_key = experiment.optimization_metric or "f1"
        job_results = []
        for job in jobs:
            metrics = job.metrics or {}
            job_results.append({
                "job_id": job.id,
                "model_type": job.model_type,
                "metrics": metrics,
                "feature_importance": job.feature_importance,
                "hyperparameters": job.hyperparameters,
                "training_duration_seconds": job.training_duration_seconds,
                "is_best": job.id == experiment.best_job_id,
            })

        # Sort by optimization metric descending
        job_results.sort(
            key=lambda j: j["metrics"].get(metric_key, 0),
            reverse=True,
        )

        return {
            "experiment_id": experiment.id,
            "name": experiment.name,
            "dataset_id": experiment.dataset_id,
            "optimization_metric": metric_key,
            "status": experiment.status,
            "best_job_id": experiment.best_job_id,
            "jobs": job_results,
        }

    def _find_best_job(
        self, jobs: List[TrainingJob], metric_key: str
    ) -> TrainingJob:
        """Find the job with the best optimization metric."""
        # For regression metrics like rmse/mae, lower is better
        lower_is_better = metric_key in ("rmse", "mae")

        def get_metric(job):
            return (job.metrics or {}).get(metric_key, float("inf") if lower_is_better else 0)

        if lower_is_better:
            return min(jobs, key=get_metric)
        return max(jobs, key=get_metric)
