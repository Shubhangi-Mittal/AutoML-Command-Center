import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureEngine:
    """Automated feature engineering pipeline for tabular data."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type: str,
        drop_columns: Optional[List[str]] = None,
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.task_type = task_type
        self.drop_columns = drop_columns or []
        self.transformations: List[str] = []
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None

    def auto_engineer(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """Run the full feature engineering pipeline and return train/test splits."""
        self._drop_id_columns()
        self._handle_missing()
        self._encode_categoricals()
        self._handle_skewness()
        self._scale_features()

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        metadata = {
            "transformations": self.transformations,
            "feature_count": X_train.shape[1],
            "feature_names": list(X_train.columns),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        return X_train, X_test, y_train, y_test, metadata

    def _drop_id_columns(self) -> None:
        """Drop user-specified columns and auto-detected ID-like columns."""
        cols_to_drop = list(self.drop_columns)

        for col in self.df.columns:
            if col == self.target_col:
                continue
            if self.df[col].dtype == object:
                unique_pct = self.df[col].nunique() / max(len(self.df), 1)
                if unique_pct > 0.95:
                    cols_to_drop.append(col)

        cols_to_drop = [c for c in set(cols_to_drop) if c in self.df.columns]
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            self.transformations.append(f"Dropped columns: {cols_to_drop}")

    def _handle_missing(self) -> None:
        """Impute missing values: median for numeric, mode for categorical."""
        for col in self.df.columns:
            missing = self.df[col].isna().sum()
            if missing == 0:
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                self.transformations.append(
                    f"Imputed '{col}' with median ({median_val:.4f})"
                )
            else:
                mode_val = self.df[col].mode()
                if len(mode_val) > 0:
                    self.df[col].fillna(mode_val[0], inplace=True)
                    self.transformations.append(
                        f"Imputed '{col}' with mode ('{mode_val[0]}')"
                    )
                else:
                    self.df[col].fillna("unknown", inplace=True)
                    self.transformations.append(
                        f"Imputed '{col}' with 'unknown'"
                    )

    def _encode_categoricals(self) -> None:
        """One-hot encode low cardinality, label encode high cardinality."""
        cat_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        if self.target_col in cat_cols:
            cat_cols.remove(self.target_col)

        # Encode target if categorical
        if self.df[self.target_col].dtype == object:
            le = LabelEncoder()
            self.df[self.target_col] = le.fit_transform(self.df[self.target_col])
            self.label_encoders[self.target_col] = le
            self.transformations.append(
                f"Label-encoded target '{self.target_col}'"
            )

        low_cardinality = [c for c in cat_cols if self.df[c].nunique() <= 10]
        high_cardinality = [c for c in cat_cols if self.df[c].nunique() > 10]

        # One-hot encode low cardinality
        if low_cardinality:
            self.df = pd.get_dummies(self.df, columns=low_cardinality, drop_first=True)
            self.transformations.append(
                f"One-hot encoded: {low_cardinality}"
            )

        # Label encode high cardinality
        for col in high_cardinality:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            self.transformations.append(f"Label-encoded '{col}'")

    def _handle_skewness(self) -> None:
        """Log-transform highly skewed numeric features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        for col in numeric_cols:
            skewness = self.df[col].skew()
            if abs(skewness) > 2 and (self.df[col] >= 0).all():
                self.df[col] = np.log1p(self.df[col])
                self.transformations.append(
                    f"Log-transformed '{col}' (skewness={skewness:.2f})"
                )

    def _scale_features(self) -> None:
        """Standard-scale all numeric features (excluding target)."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        if numeric_cols:
            self.scaler = StandardScaler()
            self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])
            self.transformations.append(
                f"Standard-scaled {len(numeric_cols)} numeric features"
            )
