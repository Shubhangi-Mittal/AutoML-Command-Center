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
    {
        "name": "get_prediction_template",
        "description": "Generate a sample JSON template for making predictions with the deployed model. Returns example feature values based on the dataset, so the user can see what input format is needed. Call this when a user asks for sample/test/example JSON for prediction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "The dataset ID to generate the template from.",
                },
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "make_prediction",
        "description": "Make a prediction using the currently deployed model. Use this after deploying a model to test it with sample data, or when the user asks to predict/test/try the model. You can get a sample template first using get_prediction_template.",
        "input_schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "description": "Feature key-value pairs to predict on. Keys should match the dataset column names (excluding the target).",
                },
            },
            "required": ["features"],
        },
    },
    {
        "name": "get_serving_status",
        "description": "Check the current model serving/deployment status. Returns whether a model is deployed, its type, metrics, and deployment time.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "suggest_improvements",
        "description": "Analyze the current experiment results and suggest ways to improve model performance. Looks at metrics, feature importance, and data quality to give actionable recommendations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "The dataset ID to analyze.",
                },
            },
            "required": ["dataset_id"],
        },
    },
]


# --- Tool Executors ---

async def execute_profile_dataset(dataset_id: str, db: Session, **kwargs) -> Dict[str, Any]:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return {"error": f"Dataset {dataset_id} not found"}

    profile = dataset.profile
    if not profile:
        profile = profile_dataset(dataset.file_path)
        dataset.profile = profile
        dataset.rows = profile.get("row_count")
        dataset.columns = profile.get("column_count")
        dataset.target_column = dataset.target_column or profile.get("suggested_target")
        dataset.task_type = dataset.task_type or profile.get("suggested_task_type")
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

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


async def execute_get_prediction_template(dataset_id: str, db: Session, **kwargs) -> Dict[str, Any]:
    """Generate a sample prediction JSON from a dataset's first row (excluding target)."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return {"error": f"Dataset {dataset_id} not found"}

    df = pd.read_csv(dataset.file_path, encoding="utf-8-sig", nrows=1)
    target_col = dataset.target_column

    # Exclude target column from features
    feature_cols = [c for c in df.columns if c != target_col]
    sample_row = df[feature_cols].iloc[0].to_dict()

    # Clean up NaN values for JSON
    for k, v in sample_row.items():
        if pd.isna(v):
            sample_row[k] = None

    return {
        "dataset_name": dataset.name,
        "target_column": target_col,
        "task_type": dataset.task_type,
        "feature_columns": feature_cols,
        "sample_input": sample_row,
        "hint": "Use these feature names and similar values to make predictions. The target column is excluded from input.",
    }


async def execute_make_prediction(features: dict, db: Session = None, **kwargs) -> Dict[str, Any]:
    """Make a prediction using the currently deployed model."""
    server = ModelServer.get_instance()
    status = server.get_status()

    if status["status"] != "deployed":
        return {"error": "No model is currently deployed. Deploy a model first using deploy_model."}

    try:
        result = server.predict(features)
        result["target_column"] = None

        # Try to find the target column name from the job's dataset
        if db and status.get("job_id"):
            job = db.query(TrainingJob).filter(TrainingJob.id == status["job_id"]).first()
            if job:
                ds = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
                if ds:
                    result["target_column"] = ds.target_column

        return result
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


async def execute_get_serving_status(db: Session = None, **kwargs) -> Dict[str, Any]:
    """Check current model serving status."""
    server = ModelServer.get_instance()
    status = server.get_status()

    if status["status"] == "deployed" and db and status.get("job_id"):
        job = db.query(TrainingJob).filter(TrainingJob.id == status["job_id"]).first()
        if job:
            ds = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
            if ds:
                status["dataset_name"] = ds.name
                status["target_column"] = ds.target_column

    return status


async def execute_suggest_improvements(dataset_id: str, db: Session, **kwargs) -> Dict[str, Any]:
    """Analyze experiments and suggest improvements."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return {"error": f"Dataset {dataset_id} not found"}

    # Get latest experiment
    experiment = db.query(Experiment).filter(
        Experiment.dataset_id == dataset_id
    ).order_by(Experiment.created_at.desc()).first()

    if not experiment:
        return {"error": "No experiments found. Train models first."}

    jobs = db.query(TrainingJob).filter(
        TrainingJob.experiment_id == experiment.id,
        TrainingJob.status == "completed",
    ).all()

    if not jobs:
        return {"error": "No completed jobs found."}

    # Analyze profile for data quality
    suggestions = []
    profile = dataset.profile or {}
    warnings = profile.get("warnings", [])

    # Data quality suggestions
    for w in warnings:
        if "missing" in w.lower():
            suggestions.append(f"Data quality: {w} — Consider domain-specific imputation or dropping rows with too many missing values.")
        elif "skew" in w.lower():
            suggestions.append(f"Data quality: {w} — Already handled by auto feature engineering, but consider binning or winsorizing extreme outliers.")

    # Model performance suggestions
    best_job = None
    for j in jobs:
        if j.id == experiment.best_job_id:
            best_job = j
            break

    if best_job and best_job.metrics:
        metrics = best_job.metrics
        if "f1" in metrics and metrics["f1"] < 0.7:
            suggestions.append("F1 score is below 0.7 — try collecting more data, engineering interaction features, or tuning hyperparameters.")
        if "accuracy" in metrics and metrics["accuracy"] < 0.8:
            suggestions.append("Accuracy is below 80% — check for class imbalance and consider oversampling (SMOTE) or class weights.")
        if "r2" in metrics and metrics["r2"] < 0.5:
            suggestions.append("R² is below 0.5 — the model explains less than half the variance. Consider adding polynomial features or more relevant predictors.")

    # Feature importance suggestions
    if best_job and best_job.feature_importance:
        fi = best_job.feature_importance
        total_features = len(fi)
        low_importance = [k for k, v in fi.items() if v < 0.01]
        if len(low_importance) > total_features * 0.5:
            suggestions.append(f"{len(low_importance)} of {total_features} features have near-zero importance — consider removing them to reduce noise and overfitting.")

        top_features = list(fi.items())[:3]
        if top_features:
            suggestions.append(f"Top predictive features: {', '.join(f'{k} ({v:.3f})' for k, v in top_features)}. Consider engineering more features from these.")

    # General suggestions
    model_types_tried = [j.model_type for j in jobs]
    if len(model_types_tried) < 3:
        missing = set(["linear", "xgboost", "random_forest"]) - set(model_types_tried)
        if missing:
            suggestions.append(f"Haven't tried: {', '.join(missing)}. Training more model types could find a better fit.")

    if not suggestions:
        suggestions.append("Model performance looks good! Consider cross-validation for more robust estimates.")

    return {
        "dataset_name": dataset.name,
        "best_model": best_job.model_type if best_job else None,
        "best_metrics": best_job.metrics if best_job else None,
        "suggestions": suggestions,
        "models_trained": len(jobs),
    }


# Map tool names to executors
TOOL_EXECUTORS = {
    "profile_dataset": execute_profile_dataset,
    "get_dataset_sample": execute_get_dataset_sample,
    "launch_training": execute_launch_training,
    "query_experiments": execute_query_experiments,
    "deploy_model": execute_deploy_model,
    "get_prediction_template": execute_get_prediction_template,
    "make_prediction": execute_make_prediction,
    "get_serving_status": execute_get_serving_status,
    "suggest_improvements": execute_suggest_improvements,
}
