from __future__ import annotations

import io
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


FEATURE_NAMES = [
    "Women's Empowerment Group - 2022",
    "Global Gender Parity Index (GGPI) - 2022",
    "Gender Parity Group - 2022",
    "Human Development Group - 2021",
    "Sustainable Development Goal regions",
]

TARGET_NAME = "Women's Empowerment Index (WEI) - 2022"

CATEGORICAL_COLS = [
    "Women's Empowerment Group - 2022",
    "Gender Parity Group - 2022",
    "Human Development Group - 2021",
    "Sustainable Development Goal regions",
]


def _project_paths() -> Tuple[Path, Path]:
    """
    Returns:
      (api_dir, linear_regression_dir)
    """
    api_dir = Path(__file__).resolve().parent
    linear_regression_dir = api_dir.parent / "linear_regression"
    return api_dir, linear_regression_dir


def _artifact_paths() -> Dict[str, Path]:
    api_dir, linear_regression_dir = _project_paths()

    return {
        "best_model_api": api_dir / "best_model.pkl",
        "scaler_api": api_dir / "scaler.pkl",
        "encoders_api": api_dir / "label_encoders.pkl",
        "dataset_csv": linear_regression_dir / "women_empowerment_index.csv",
        "best_model_fallback": linear_regression_dir / "best_model.pkl",
        "scaler_fallback": linear_regression_dir / "scaler.pkl",
    }


def _fit_label_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        encoders[col] = le.fit(df[col].astype(str).fillna(""))
    return encoders


def _encode_features(
    df_features: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
) -> pd.DataFrame:
    df_encoded = df_features.copy()
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        # LabelEncoder can throw on unknown values; we surface a helpful error upstream.
        df_encoded[col] = le.transform(df_encoded[col].astype(str).fillna(""))
    return df_encoded


def load_artifacts() -> Tuple[Any, StandardScaler, Dict[str, LabelEncoder]]:
    paths = _artifact_paths()

    # Model/scaler: prefer retrained artifacts from API folder, fallback to notebook artifacts.
    best_model_path = paths["best_model_api"] if paths["best_model_api"].exists() else paths["best_model_fallback"]
    scaler_path = paths["scaler_api"] if paths["scaler_api"].exists() else paths["scaler_fallback"]

    best_model = joblib.load(best_model_path)
    scaler: StandardScaler = joblib.load(scaler_path)

    # Encoders: if not saved yet, fit them using the original dataset so labels match training.
    if paths["encoders_api"].exists():
        encoders = joblib.load(paths["encoders_api"])
    else:
        df = pd.read_csv(paths["dataset_csv"])
        encoders = _fit_label_encoders(df)

    return best_model, scaler, encoders


def predict_wei(
    *,
    women_empowerment_group: str,
    ggpi: float,
    gender_parity_group: str,
    human_development_group: str,
    sdd_regions: str,
) -> float:
    """
    Make a prediction using the best performing saved model.

    Note: Categorical values are label-encoded to match the training pipeline.
    """
    model, scaler, encoders = load_artifacts()

    feature_row = {
        "Women's Empowerment Group - 2022": women_empowerment_group,
        "Global Gender Parity Index (GGPI) - 2022": ggpi,
        "Gender Parity Group - 2022": gender_parity_group,
        "Human Development Group - 2021": human_development_group,
        "Sustainable Development Goal regions": sdd_regions,
    }

    df_features = pd.DataFrame([feature_row], columns=FEATURE_NAMES)

    # Encode categorical columns.
    df_encoded = _encode_features(df_features, encoders)

    # Scale and predict.
    x_scaled = scaler.transform(df_encoded[FEATURE_NAMES])
    pred = model.predict(x_scaled)
    return float(pred[0])


def _validate_dataset_columns(df: pd.DataFrame) -> None:
    required = set([TARGET_NAME] + FEATURE_NAMES + ["Country"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Uploaded dataset is missing required columns: {missing}")


def train_best_model_from_df(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, StandardScaler, Dict[str, LabelEncoder], Dict[str, Any]]:
    """
    Trains LinearRegression, DecisionTreeRegressor, RandomForestRegressor and selects the best by test MSE.

    Returns:
      (best_model, scaler, encoders, metrics)
    """
    _validate_dataset_columns(df)

    # Prepare X/y.
    df = df.copy()
    df = df.drop(columns=["Country"])

    y = df[TARGET_NAME].astype(float)
    X = df[FEATURE_NAMES].copy()

    # Fit encoders on the categorical columns.
    encoders = _fit_label_encoders(X.assign(_dummy=df["Country"] if "Country" in df.columns else ""))

    # Encode and split.
    X_encoded = _encode_features(X[FEATURE_NAMES], encoders)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=random_state),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=random_state),
    }

    scores: Dict[str, Any] = {}
    best_name = None
    best_mse = float("inf")
    best_model = None

    for name, mdl in models.items():
        mdl.fit(X_train_scaled, y_train)
        pred_test = mdl.predict(X_test_scaled)
        mse = mean_squared_error(y_test, pred_test)
        r2 = r2_score(y_test, pred_test)
        scores[name] = {"mse": float(mse), "r2": float(r2)}

        if mse < best_mse:
            best_mse = mse
            best_name = name
            best_model = mdl

    metrics = {
        "best_model": best_name,
        "best_mse": float(best_mse),
        "scores": scores,
    }

    assert best_model is not None
    return best_model, scaler, encoders, metrics


def retrain_and_save(
    *,
    csv_bytes: bytes,
) -> Dict[str, Any]:
    """
    Retrains the model from the provided CSV content and saves artifacts in the API folder.
    """
    df = pd.read_csv(io.BytesIO(csv_bytes))
    best_model, scaler, encoders, metrics = train_best_model_from_df(df)

    paths = _artifact_paths()
    joblib.dump(best_model, paths["best_model_api"])
    joblib.dump(scaler, paths["scaler_api"])
    joblib.dump(encoders, paths["encoders_api"])

    return metrics

