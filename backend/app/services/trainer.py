import os
import time
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
) -> Dict[str, Any]:
    """Train a single model and return results with metrics."""
    hyperparameters = hyperparameters or {}
    start_time = time.time()

    if model_type == "linear":
        result = _train_linear(X_train, X_test, y_train, y_test, task_type, hyperparameters)
    elif model_type == "xgboost":
        result = _train_xgboost(X_train, X_test, y_train, y_test, task_type, hyperparameters)
    elif model_type == "random_forest":
        result = _train_random_forest(X_train, X_test, y_train, y_test, task_type, hyperparameters)
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
) -> List[Dict[str, Any]]:
    """Train multiple model types and return all results."""
    if model_types is None:
        model_types = ["linear", "xgboost", "random_forest"]

    results = []
    for model_type in model_types:
        result = train_model(X_train, X_test, y_train, y_test, model_type, task_type)
        results.append(result)

    return results


def _train_linear(
    X_train, X_test, y_train, y_test, task_type, hyperparameters
) -> Dict[str, Any]:
    """Train a linear model (LogisticRegression or Ridge)."""
    if task_type == "classification":
        model = LogisticRegression(max_iter=1000, **hyperparameters)
    else:
        model = Ridge(**hyperparameters)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred, task_type)

    feature_importance = _get_linear_importance(model, X_train.columns)
    model_path = _save_model(model, "linear")

    return {
        "model_type": "linear",
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model_path": model_path,
        "hyperparameters": hyperparameters,
    }


def _train_xgboost(
    X_train, X_test, y_train, y_test, task_type, hyperparameters
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

    early_stopping = defaults.pop("early_stopping_rounds")
    eval_metric = defaults.pop("eval_metric")

    if task_type == "classification":
        model = XGBClassifier(**defaults, eval_metric=eval_metric)
    else:
        model = XGBRegressor(**defaults, eval_metric=eval_metric)

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

    model_path = _save_model(model, "xgboost")

    return {
        "model_type": "xgboost",
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model_path": model_path,
        "hyperparameters": defaults,
    }


def _train_random_forest(
    X_train, X_test, y_train, y_test, task_type, hyperparameters
) -> Dict[str, Any]:
    """Train a Random Forest model."""
    defaults = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    defaults.update(hyperparameters)

    if task_type == "classification":
        model = RandomForestClassifier(**defaults)
    else:
        model = RandomForestRegressor(**defaults)

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

    model_path = _save_model(model, "random_forest")

    return {
        "model_type": "random_forest",
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model_path": model_path,
        "hyperparameters": defaults,
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


def _save_model(model: Any, model_type: str) -> str:
    """Serialize model to disk and return the file path."""
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(
        settings.MODEL_DIR, f"{model_type}_{int(time.time())}.pkl"
    )
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model_path
