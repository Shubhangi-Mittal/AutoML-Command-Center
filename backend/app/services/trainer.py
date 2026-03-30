import os
import time
import pickle
from itertools import product
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from xgboost import XGBClassifier, XGBRegressor
from app.config import settings


def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str,
    task_type: str,
    hyperparameters: Dict[str, Any] = None,
    artifact_id: str = None,
    optimization_metric: str = None,
    cv_folds: int = 1,
    tune_hyperparameters: bool = False,
) -> Dict[str, Any]:
    """Train a single model and return results with metrics."""
    hyperparameters = hyperparameters or {}
    start_time = time.time()

    optimization_metric = optimization_metric or ("f1" if task_type == "classification" else "rmse")
    if model_type == "linear":
        result = _train_linear(
            X_train, X_test, y_train, y_test, task_type, hyperparameters,
            artifact_id, optimization_metric, cv_folds, tune_hyperparameters,
        )
    elif model_type == "xgboost":
        result = _train_xgboost(
            X_train, X_test, y_train, y_test, task_type, hyperparameters,
            artifact_id, optimization_metric, cv_folds, tune_hyperparameters,
        )
    elif model_type == "random_forest":
        result = _train_random_forest(
            X_train, X_test, y_train, y_test, task_type, hyperparameters,
            artifact_id, optimization_metric, cv_folds, tune_hyperparameters,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    result["training_duration_seconds"] = round(time.time() - start_time, 3)
    return result


def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    model_types: List[str] = None,
    artifact_ids: Dict[str, str] = None,
    optimization_metric: str = None,
    cv_folds: int = 1,
    tune_hyperparameters: bool = False,
) -> List[Dict[str, Any]]:
    """Train multiple model types and return all results."""
    if model_types is None:
        model_types = ["linear", "xgboost", "random_forest"]

    results = []
    for model_type in model_types:
        result = train_model(
            X_train,
            X_test,
            y_train,
            y_test,
            model_type,
            task_type,
            artifact_id=(artifact_ids or {}).get(model_type),
            optimization_metric=optimization_metric,
            cv_folds=cv_folds,
            tune_hyperparameters=tune_hyperparameters,
        )
        results.append(result)

    return results


def _train_linear(
    X_train, X_test, y_train, y_test, task_type, hyperparameters, artifact_id,
    optimization_metric, cv_folds, tune_hyperparameters,
) -> Dict[str, Any]:
    """Train a linear model (LogisticRegression or Ridge)."""
    hyperparameters = hyperparameters or {}
    if tune_hyperparameters:
        hyperparameters = _tune_hyperparameters(
            "linear", task_type, X_train, X_test, y_train, y_test, optimization_metric, hyperparameters
        )

    model = _create_estimator("linear", task_type, hyperparameters)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred, task_type)

    feature_importance = _get_linear_importance(model, X_train.columns)
    model_path = _save_model(model, "linear", list(X_train.columns), artifact_id=artifact_id)

    return {
        "model_type": "linear",
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model_path": model_path,
        "hyperparameters": hyperparameters,
        "cross_validation": _compute_cross_validation(model, X_train, y_train, task_type, cv_folds),
    }


def _train_xgboost(
    X_train, X_test, y_train, y_test, task_type, hyperparameters, artifact_id,
    optimization_metric, cv_folds, tune_hyperparameters,
) -> Dict[str, Any]:
    """Train an XGBoost model with early stopping."""
    defaults = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "early_stopping_rounds": 20,
        "eval_metric": "logloss" if task_type == "classification" else "rmse",
    }
    defaults.update(hyperparameters)

    if tune_hyperparameters:
        defaults = _tune_hyperparameters(
            "xgboost", task_type, X_train, X_test, y_train, y_test, optimization_metric, defaults
        )

    early_stopping = defaults.pop("early_stopping_rounds")
    eval_metric = defaults.pop("eval_metric")

    model = _create_estimator("xgboost", task_type, defaults, eval_metric=eval_metric)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred, task_type)

    importance = model.feature_importances_
    feature_importance = {
        col: round(float(imp), 4)
        for col, imp in zip(X_train.columns, importance)
    }
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    model_path = _save_model(model, "xgboost", list(X_train.columns), artifact_id=artifact_id)

    return {
        "model_type": "xgboost",
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model_path": model_path,
        "hyperparameters": defaults,
        "cross_validation": _compute_cross_validation(model, X_train, y_train, task_type, cv_folds),
    }


def _train_random_forest(
    X_train, X_test, y_train, y_test, task_type, hyperparameters, artifact_id,
    optimization_metric, cv_folds, tune_hyperparameters,
) -> Dict[str, Any]:
    """Train a Random Forest model."""
    defaults = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    defaults.update(hyperparameters)

    if tune_hyperparameters:
        defaults = _tune_hyperparameters(
            "random_forest", task_type, X_train, X_test, y_train, y_test, optimization_metric, defaults
        )

    model = _create_estimator("random_forest", task_type, defaults)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred, task_type)

    importance = model.feature_importances_
    feature_importance = {
        col: round(float(imp), 4)
        for col, imp in zip(X_train.columns, importance)
    }
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    model_path = _save_model(model, "random_forest", list(X_train.columns), artifact_id=artifact_id)

    return {
        "model_type": "random_forest",
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model_path": model_path,
        "hyperparameters": defaults,
        "cross_validation": _compute_cross_validation(model, X_train, y_train, task_type, cv_folds),
    }


def _compute_metrics(y_test, y_pred, task_type: str) -> Dict[str, Any]:
    """Compute classification or regression metrics."""
    if task_type == "classification":
        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
            "precision": round(float(precision_score(y_test, y_pred, average="weighted")), 4),
            "recall": round(float(recall_score(y_test, y_pred, average="weighted")), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        return {
            "rmse": round(rmse, 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "r2": round(float(r2_score(y_test, y_pred)), 4),
        }


def _get_linear_importance(model, feature_names) -> Dict[str, float]:
    """Extract feature importance from linear model coefficients."""
    if hasattr(model, "coef_"):
        coefs = model.coef_
        if coefs.ndim > 1:
            coefs = np.abs(coefs).mean(axis=0)
        else:
            coefs = np.abs(coefs)
        importance = {
            col: round(float(imp), 4)
            for col, imp in zip(feature_names, coefs)
        }
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return {}


def _create_estimator(model_type: str, task_type: str, hyperparameters: Dict[str, Any], eval_metric: str = None):
    if model_type == "linear":
        if task_type == "classification":
            return LogisticRegression(max_iter=1000, **hyperparameters)
        return Ridge(**hyperparameters)

    if model_type == "xgboost":
        if task_type == "classification":
            return XGBClassifier(**hyperparameters, eval_metric=eval_metric or "logloss")
        return XGBRegressor(**hyperparameters, eval_metric=eval_metric or "rmse")

    if model_type == "random_forest":
        if task_type == "classification":
            return RandomForestClassifier(**hyperparameters)
        return RandomForestRegressor(**hyperparameters)

    raise ValueError(f"Unknown model type: {model_type}")


def _compute_cross_validation(model, X, y, task_type: str, cv_folds: int) -> Dict[str, Any] | None:
    if cv_folds <= 1 or len(X) < cv_folds:
        return None

    scoring = (
        ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
        if task_type == "classification"
        else ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
    )
    splitter = (
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        if task_type == "classification"
        else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    )
    scores = cross_validate(clone(model), X, y, cv=splitter, scoring=scoring)

    summary = {}
    for metric_name in scoring:
        values = scores[f"test_{metric_name}"]
        clean_name = metric_name.replace("neg_", "").replace("_weighted", "")
        if metric_name.startswith("neg_"):
            values = -values
        summary[clean_name] = round(float(np.mean(values)), 4)
        summary[f"{clean_name}_std"] = round(float(np.std(values)), 4)
    return summary


def _tune_hyperparameters(
    model_type: str,
    task_type: str,
    X_train,
    X_test,
    y_train,
    y_test,
    optimization_metric: str,
    base_params: Dict[str, Any],
) -> Dict[str, Any]:
    candidates = _candidate_hyperparameters(model_type, task_type, base_params)
    best_params = dict(base_params)
    best_score = None

    for params in candidates:
        model = _create_estimator(
            model_type,
            task_type,
            params,
            eval_metric="logloss" if task_type == "classification" else "rmse",
        )
        if model_type == "xgboost":
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = _compute_metrics(y_test, y_pred, task_type)
        score = metrics.get(optimization_metric)
        if score is None:
            continue
        if _is_better_score(score, best_score, optimization_metric):
            best_score = score
            best_params = params

    return best_params


def _candidate_hyperparameters(
    model_type: str,
    task_type: str,
    base_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    grid = {}
    if model_type == "linear":
        grid = {"C": [0.1, 1.0, 5.0]} if task_type == "classification" else {"alpha": [0.1, 1.0, 10.0]}
    elif model_type == "xgboost":
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
        }
    elif model_type == "random_forest":
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [8, 12],
        }

    if not grid:
        return [dict(base_params)]

    keys = list(grid.keys())
    combos = []
    for values in product(*(grid[key] for key in keys)):
        params = dict(base_params)
        params.update(dict(zip(keys, values)))
        combos.append(params)
    return combos[:6]


def _is_better_score(score: float, best_score: float | None, metric_name: str) -> bool:
    if best_score is None:
        return True
    if metric_name in ("rmse", "mae"):
        return score < best_score
    return score > best_score


def _save_model(
    model: Any,
    model_type: str,
    feature_names: list = None,
    artifact_id: str = None,
) -> str:
    """Serialize model and feature names to disk and return the file path."""
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    filename_parts = [model_type, str(int(time.time()))]
    if artifact_id:
        filename_parts.append(artifact_id)
    model_path = os.path.join(settings.MODEL_DIR, f"{'_'.join(filename_parts)}.pkl")
    bundle = {"model": model, "feature_names": feature_names or []}
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    return model_path


def finalize_model_artifact(model_path: str, model_type: str, artifact_id: str) -> str:
    """Rename an existing artifact so it can be resolved deterministically later."""
    if not model_path or not os.path.exists(model_path) or not artifact_id:
        return model_path

    target_path = os.path.join(settings.MODEL_DIR, f"{model_type}_{artifact_id}.pkl")
    if model_path == target_path:
        return model_path

    os.replace(model_path, target_path)
    return target_path


def resolve_model_artifact(model_type: str, artifact_id: str = None) -> str | None:
    """Resolve a model artifact, preferring an exact job-specific file when available."""
    if not os.path.isdir(settings.MODEL_DIR):
        return None

    if artifact_id:
        exact_path = os.path.join(settings.MODEL_DIR, f"{model_type}_{artifact_id}.pkl")
        if os.path.exists(exact_path):
            return exact_path

    prefix = f"{model_type}_"
    candidates = [
        os.path.join(settings.MODEL_DIR, name)
        for name in os.listdir(settings.MODEL_DIR)
        if name.startswith(prefix) and name.endswith(".pkl")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)
