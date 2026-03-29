"""Tool definitions and executors for the AI agent."""

import pandas as pd
from typing import Any, Dict
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.job import TrainingJob
from app.models.experiment import Experiment
from app.services.profiler import profile_dataset
from app.services.feature_engine import FeatureEngine
from app.services.trainer import train_all_models
from app.services.experiment_tracker import ExperimentTracker
from app.services.serving import ModelServer


# --- Claude API Tool Definitions ---

TOOL_DEFINITIONS = [
    {
        "name": "profile_dataset",
        "description": "Analyze an uploaded dataset. Returns column statistics, correlations, data quality warnings, and suggested target/task type. Always call this first before training.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "The ID of the dataset to profile",
                },
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "get_dataset_sample",
        "description": "Get sample rows from the dataset to inspect actual data values. Useful for understanding the data before making decisions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "The ID of the dataset",
                },
                "n_rows": {
                    "type": "integer",
                    "description": "Number of sample rows to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "launch_training",
        "description": "Train multiple model types on a dataset. Automatically handles feature engineering (missing values, encoding, scaling). Returns metrics and feature importance for each model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "The ID of the dataset to train on",
                },
                "target_column": {
                    "type": "string",
                    "description": "The target column name. If omitted, uses the suggested target from profiling.",
                },
                "task_type": {
                    "type": "string",
                    "enum": ["classification", "regression"],
                    "description": "The ML task type. If omitted, uses the inferred type.",
                },
                "model_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["linear", "xgboost", "random_forest"]},
                    "description": "Which models to train. Default: all three.",
                },
                "optimization_metric": {
                    "type": "string",
                    "description": "Metric to optimize for (e.g., 'f1', 'accuracy', 'recall', 'rmse'). Default: 'f1' for classification, 'rmse' for regression.",
                },
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "query_experiments",
        "description": "Query completed experiment results. Returns model comparisons sorted by the optimization metric, with feature importance and the best model highlighted.",
        "input_schema": {
            "type": "object",
            "properties": {
                "experiment_id": {
                    "type": "string",
                    "description": "The experiment ID to query. If omitted, returns the latest experiment for the dataset.",
                },
                "dataset_id": {
                    "type": "string",
                    "description": "The dataset ID to find experiments for.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "deploy_model",
        "description": "Deploy a trained model to the REST API for serving predictions. Uses the best model from an experiment or a specific job ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The training job ID to deploy. If omitted, deploys the best model from the latest experiment.",
                },
                "dataset_id": {
                    "type": "string",
                    "description": "The dataset ID to find the best model for.",
                },
            },
            "required": [],
        },
    },
]


# --- Tool Executors ---

async def execute_profile_dataset(dataset_id: str, db: Session, **kwargs) -> Dict[str, Any]:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return {"error": f"Dataset {dataset_id} not found"}

    profile = profile_dataset(dataset.file_path)
    return {
        "dataset_name": dataset.name,
        "row_count": profile["row_count"],
        "column_count": profile["column_count"],
        "memory_usage_mb": profile["memory_usage_mb"],
        "duplicate_rows": profile["duplicate_rows"],
        "suggested_target": profile["suggested_target"],
        "suggested_task_type": profile["suggested_task_type"],
        "warnings": profile["warnings"],
        "columns": {
            col: {
                "type": info.get("dtype_category", info.get("type")),
                "missing_pct": info["missing_pct"],
                "unique_count": info["unique_count"],
                "mean": info.get("mean"),
                "std": info.get("std"),
                "top_values": info.get("top_values"),
            }
            for col, info in profile["columns"].items()
        },
        "top_correlations": profile["correlations"].get("top_pairs", [])[:5],
    }


async def execute_get_dataset_sample(dataset_id: str, db: Session, n_rows: int = 5, **kwargs) -> Dict[str, Any]:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return {"error": f"Dataset {dataset_id} not found"}

    df = pd.read_csv(dataset.file_path, encoding="utf-8-sig", nrows=n_rows)
    return {
        "dataset_name": dataset.name,
        "sample": df.to_dict(orient="records"),
        "columns": list(df.columns),
    }


async def execute_launch_training(
    dataset_id: str, db: Session,
    target_column: str = None, task_type: str = None,
    model_types: list = None, optimization_metric: str = None,
    **kwargs,
) -> Dict[str, Any]:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return {"error": f"Dataset {dataset_id} not found"}

    target_col = target_column or dataset.target_column
    t_type = task_type or dataset.task_type
    if not target_col or not t_type:
        return {"error": "Cannot determine target column or task type. Please specify."}

    m_types = model_types or ["linear", "xgboost", "random_forest"]
    opt_metric = optimization_metric or ("f1" if t_type == "classification" else "rmse")

    # Create experiment
    tracker = ExperimentTracker(db)
    experiment = tracker.create_experiment(
        dataset_id=dataset.id,
        name=f"{dataset.name} - {', '.join(m_types)}",
        optimization_metric=opt_metric,
    )

    # Feature engineering + training
    df = pd.read_csv(dataset.file_path, encoding="utf-8-sig")
    engine = FeatureEngine(df, target_col, t_type)
    X_train, X_test, y_train, y_test, metadata = engine.auto_engineer()

    results = train_all_models(X_train, X_test, y_train, y_test, t_type, m_types)

    # Save jobs
    jobs_summary = []
    for result in results:
        job = TrainingJob(
            dataset_id=dataset.id,
            experiment_id=experiment.id,
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

        jobs_summary.append({
            "job_id": job.id,
            "model_type": result["model_type"],
            "metrics": result["metrics"],
            "top_features": dict(list(result.get("feature_importance", {}).items())[:5]),
            "training_duration_seconds": result["training_duration_seconds"],
        })

    # Complete experiment
    tracker.complete_experiment(experiment.id)
    experiment = db.query(Experiment).filter(Experiment.id == experiment.id).first()

    return {
        "experiment_id": experiment.id,
        "optimization_metric": opt_metric,
        "best_job_id": experiment.best_job_id,
        "feature_engineering": {
            "transformations": metadata["transformations"],
            "feature_count": metadata["feature_count"],
        },
        "jobs": jobs_summary,
    }


async def execute_query_experiments(
    db: Session, experiment_id: str = None, dataset_id: str = None, **kwargs
) -> Dict[str, Any]:
    if experiment_id:
        tracker = ExperimentTracker(db)
        return tracker.compare_jobs(experiment_id)

    # Find latest experiment for dataset
    query = db.query(Experiment)
    if dataset_id:
        query = query.filter(Experiment.dataset_id == dataset_id)
    experiment = query.order_by(Experiment.created_at.desc()).first()

    if not experiment:
        return {"error": "No experiments found"}

    tracker = ExperimentTracker(db)
    return tracker.compare_jobs(experiment.id)


async def execute_deploy_model(
    db: Session, job_id: str = None, dataset_id: str = None, **kwargs
) -> Dict[str, Any]:
    if not job_id:
        # Find best job from latest experiment
        query = db.query(Experiment)
        if dataset_id:
            query = query.filter(Experiment.dataset_id == dataset_id)
        experiment = query.order_by(Experiment.created_at.desc()).first()
        if not experiment or not experiment.best_job_id:
            return {"error": "No completed experiment found to deploy from"}
        job_id = experiment.best_job_id

    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        return {"error": f"Job {job_id} not found"}

    import os, glob
    from app.config import settings

    pattern = os.path.join(settings.MODEL_DIR, f"{job.model_type}_*.pkl")
    model_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not model_files:
        return {"error": f"No model file found for {job.model_type}"}

    server = ModelServer.get_instance()
    result = server.deploy(model_files[0], {
        "job_id": job.id,
        "model_type": job.model_type,
        "metrics": job.metrics,
    })

    return {
        "status": "deployed",
        "model_type": job.model_type,
        "job_id": job.id,
        "metrics": job.metrics,
        "deployed_at": result["deployed_at"],
    }


# Map tool names to executors
TOOL_EXECUTORS = {
    "profile_dataset": execute_profile_dataset,
    "get_dataset_sample": execute_get_dataset_sample,
    "launch_training": execute_launch_training,
    "query_experiments": execute_query_experiments,
    "deploy_model": execute_deploy_model,
}
