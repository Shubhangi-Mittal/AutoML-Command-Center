import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional


# Common target column names to look for
TARGET_NAMES = [
    "target", "label", "class", "y", "output",
    "churn", "survived", "price", "salary", "revenue",
    "fraud", "default", "diagnosis", "species",
]


def profile_dataset(file_path: str) -> Dict[str, Any]:
    """Produce a comprehensive profile for a CSV dataset."""
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    profile: Dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {},
        "warnings": [],
        "suggested_target": None,
        "suggested_task_type": None,
    }

    for col in df.columns:
        profile["columns"][col] = _profile_column(df[col])

    profile["correlations"] = _compute_correlations(df)
    profile["suggested_target"] = _suggest_target(df, profile["columns"])

    if profile["suggested_target"]:
        profile["suggested_task_type"] = _infer_task_type(
            df, profile["suggested_target"], profile["columns"]
        )

    profile["warnings"] = _generate_warnings(profile)

    return profile


def _profile_column(series: pd.Series) -> Dict[str, Any]:
    """Profile a single column with comprehensive statistics."""
    col_profile: Dict[str, Any] = {
        "dtype": str(series.dtype),
        "missing_count": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean()) * 100, 2),
        "unique_count": int(series.nunique()),
        "unique_pct": round(float(series.nunique() / max(len(series), 1)) * 100, 2),
    }

    non_null = series.dropna()

    if pd.api.types.is_numeric_dtype(series) and len(non_null) > 0:
        col_profile["type"] = "numeric"
        col_profile["mean"] = round(float(non_null.mean()), 4)
        col_profile["median"] = round(float(non_null.median()), 4)
        col_profile["std"] = round(float(non_null.std()), 4)
        col_profile["min"] = float(non_null.min())
        col_profile["max"] = float(non_null.max())
        col_profile["q25"] = float(non_null.quantile(0.25))
        col_profile["q50"] = float(non_null.quantile(0.50))
        col_profile["q75"] = float(non_null.quantile(0.75))
        col_profile["skewness"] = round(float(non_null.skew()), 4)
        col_profile["kurtosis"] = round(float(non_null.kurtosis()), 4)
        col_profile["histogram"] = _compute_histogram(non_null)
    else:
        col_profile["type"] = "categorical"
        top_values = non_null.value_counts().head(10)
        col_profile["top_values"] = {
            str(k): int(v) for k, v in top_values.items()
        }
        if non_null.dtype == object:
            str_lengths = non_null.astype(str).str.len()
            col_profile["avg_string_length"] = round(float(str_lengths.mean()), 2)

    return col_profile


def _compute_histogram(series: pd.Series, bins: int = 20) -> List[Dict[str, Any]]:
    """Compute a histogram for numeric data."""
    counts, bin_edges = np.histogram(series, bins=bins)
    return [
        {
            "bin_start": round(float(bin_edges[i]), 4),
            "bin_end": round(float(bin_edges[i + 1]), 4),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]


def _compute_correlations(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute pairwise correlations for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return {}

    corr_matrix = numeric_df.corr()
    correlations: Dict[str, Dict[str, float]] = {}

    for col in corr_matrix.columns:
        correlations[col] = {
            other: round(float(corr_matrix.loc[col, other]), 4)
            for other in corr_matrix.columns
            if other != col
        }

    return correlations


def _suggest_target(
    df: pd.DataFrame, columns_profile: Dict[str, Any]
) -> Optional[str]:
    """Suggest the most likely target column."""
    col_names_lower = {col.lower(): col for col in df.columns}

    # Check for common target column names
    for target_name in TARGET_NAMES:
        if target_name in col_names_lower:
            return col_names_lower[target_name]

    # Fallback: return the last column
    if len(df.columns) > 0:
        return str(df.columns[-1])

    return None


def _infer_task_type(
    df: pd.DataFrame, target_col: str, columns_profile: Dict[str, Any]
) -> str:
    """Infer whether the target is classification or regression."""
    col_info = columns_profile.get(target_col, {})

    if col_info.get("type") == "categorical":
        return "classification"

    unique_count = col_info.get("unique_count", 0)
    row_count = len(df)

    # Low unique count relative to rows suggests classification
    if unique_count <= 20 or (row_count > 0 and unique_count / row_count < 0.05):
        return "classification"

    return "regression"


def _generate_warnings(profile: Dict[str, Any]) -> List[str]:
    """Generate actionable warnings based on the profile."""
    warnings: List[str] = []

    for col_name, col_info in profile["columns"].items():
        # High missing values
        if col_info["missing_pct"] > 5:
            warnings.append(
                f"Column '{col_name}' has {col_info['missing_pct']}% missing values"
            )

        # ID-like columns (>95% unique categorical)
        if (
            col_info.get("type") == "categorical"
            and col_info["unique_pct"] > 95
        ):
            warnings.append(
                f"Column '{col_name}' looks like an ID column "
                f"({col_info['unique_pct']}% unique values)"
            )

        # Highly skewed numeric distributions
        if col_info.get("type") == "numeric":
            skewness = col_info.get("skewness", 0)
            if abs(skewness) > 2:
                warnings.append(
                    f"Column '{col_name}' is highly skewed "
                    f"(skewness={skewness})"
                )

    # Duplicate rows
    if profile["duplicate_rows"] > 0:
        warnings.append(
            f"Dataset has {profile['duplicate_rows']} duplicate rows"
        )

    return warnings
