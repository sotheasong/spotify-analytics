from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Ordered by "most musical" signal first.
DEFAULT_AUDIO_FEATURES: list[str] = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "valence",
    "tempo",
]


@dataclass(slots=True)
class FeatureEngineeringConfig:
    user_history_csv: str = "data/raw/df_all.csv"
    catalog_csv: str = "data/raw/spotify-tracks.csv"
    feature_columns: tuple[str, ...] = tuple(DEFAULT_AUDIO_FEATURES)
    date_column: str = "collection_date"
    user_id_column: str = "id"
    catalog_id_column: str = "track_id"
    dropna_features: bool = True
    exclude_seen_tracks: bool = True
    recency_halflife_days: float = 14.0
    feature_weights: dict[str, float] | None = None


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _resolve_catalog_track_id(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    if "id" in df.columns:
        return "id"
    raise KeyError(
        "Catalog id column not found. Expected one of: "
        f"{preferred!r}, 'id'."
    )


def _resolve_user_track_id(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    if "track_id" in df.columns:
        return "track_id"
    raise KeyError(
        "User id column not found. Expected one of: "
        f"{preferred!r}, 'track_id'."
    )


def load_user_history(csv_path: str) -> pd.DataFrame:
    """Load user listening history exported from EDA/ingestion."""
    return pd.read_csv(csv_path)


def load_catalog(csv_path: str) -> pd.DataFrame:
    """Load full recommendation catalog."""
    catalog = pd.read_csv(csv_path)
    if "Unnamed: 0" in catalog.columns:
        catalog = catalog.drop(columns=["Unnamed: 0"])
    return catalog


def resolve_feature_columns(
    user_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    preferred_features: Iterable[str],
) -> list[str]:
    """Return shared, ordered feature columns available in both datasets."""
    shared = set(user_df.columns).intersection(catalog_df.columns)
    resolved = [col for col in preferred_features if col in shared]
    if not resolved:
        raise ValueError("No shared feature columns found across user and catalog data.")
    return resolved


def prepare_datasets(
    user_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    config: FeatureEngineeringConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str, str]:
    """Align schemas and clean feature values prior to scaling/modeling."""
    user = user_df.copy()
    catalog = catalog_df.copy()

    user_id_col = _resolve_user_track_id(user, config.user_id_column)
    catalog_id_col = _resolve_catalog_track_id(catalog, config.catalog_id_column)

    feature_cols = resolve_feature_columns(user, catalog, config.feature_columns)
    user = _coerce_numeric(user, feature_cols)
    catalog = _coerce_numeric(catalog, feature_cols)

    if config.dropna_features:
        user = user.dropna(subset=feature_cols)
        catalog = catalog.dropna(subset=feature_cols)
    else:
        for col in feature_cols:
            user[col] = user[col].fillna(user[col].median())
            catalog[col] = catalog[col].fillna(catalog[col].median())

    user = user.drop_duplicates(subset=[user_id_col], keep="last")
    catalog = catalog.drop_duplicates(subset=[catalog_id_col], keep="last")

    if config.exclude_seen_tracks:
        seen_ids = set(user[user_id_col].astype(str))
        catalog = catalog.loc[~catalog[catalog_id_col].astype(str).isin(seen_ids)].copy()

    return user, catalog, feature_cols, user_id_col, catalog_id_col


def scale_features(
    user_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit scaler on catalog only, then transform both.

    Fitting on the catalog keeps the scale anchored to recommendation candidates.
    """
    scaler = StandardScaler()
    catalog_scaled = scaler.fit_transform(catalog_df[feature_cols])
    user_scaled = scaler.transform(user_df[feature_cols])
    return user_scaled, catalog_scaled, scaler


def compute_recency_weights(
    user_df: pd.DataFrame,
    date_column: str,
    halflife_days: float,
) -> np.ndarray:
    """
    Compute recency-decay weights from a date column.

    If dates are missing/unparseable, fallback to uniform weights.
    """
    if date_column not in user_df.columns:
        return np.ones(len(user_df), dtype=float)

    dt = pd.to_datetime(user_df[date_column], errors="coerce")
    if dt.isna().all():
        return np.ones(len(user_df), dtype=float)

    max_date = dt.max()
    age_days = (max_date - dt).dt.days.fillna(0).astype(float)
    # Exponential decay with half-life.
    weights = np.exp(-np.log(2.0) * age_days / max(halflife_days, 1e-6))
    return weights.to_numpy(dtype=float)


def build_user_profile(
    user_scaled: np.ndarray,
    recency_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Create a single preference vector from scaled user history."""
    if user_scaled.size == 0:
        raise ValueError("user_scaled is empty; cannot build user profile.")
    if recency_weights is None:
        return user_scaled.mean(axis=0)
    w = np.asarray(recency_weights, dtype=float)
    if w.shape[0] != user_scaled.shape[0]:
        raise ValueError("recency_weights length does not match user history rows.")
    w = np.clip(w, a_min=0.0, a_max=None)
    denom = float(w.sum())
    if denom <= 0.0:
        return user_scaled.mean(axis=0)
    return np.average(user_scaled, axis=0, weights=w)


def run_feature_engineering(
    config: FeatureEngineeringConfig | None = None,
) -> dict[str, object]:
    """
    End-to-end feature engineering pipeline.

    Returns all components required for training/ranking:
    - cleaned user and catalog frames
    - shared feature columns
    - scaled matrices
    - scaler
    - recency weights
    - user profile vector
    """
    cfg = config or FeatureEngineeringConfig()
    user_df = load_user_history(str(Path(cfg.user_history_csv)))
    catalog_df = load_catalog(str(Path(cfg.catalog_csv)))

    user_clean, catalog_clean, feature_cols, user_id_col, catalog_id_col = prepare_datasets(
        user_df=user_df,
        catalog_df=catalog_df,
        config=cfg,
    )
    user_scaled, catalog_scaled, scaler = scale_features(
        user_df=user_clean,
        catalog_df=catalog_clean,
        feature_cols=feature_cols,
    )

    # Optional feature weighting (applied consistently to user and catalog vectors).
    if cfg.feature_weights:
        feature_weight_vector = np.array(
            [float(cfg.feature_weights.get(col, 1.0)) for col in feature_cols],
            dtype=float,
        )
        user_scaled = user_scaled * feature_weight_vector
        catalog_scaled = catalog_scaled * feature_weight_vector
    
    recency_weights = compute_recency_weights(
        user_df=user_clean,
        date_column=cfg.date_column,
        halflife_days=cfg.recency_halflife_days,
    )
    user_profile = build_user_profile(user_scaled=user_scaled, recency_weights=recency_weights)

    return {
        "user_df": user_clean,
        "catalog_df": catalog_clean,
        "feature_columns": feature_cols,
        "user_id_column": user_id_col,
        "catalog_id_column": catalog_id_col,
        "user_scaled": user_scaled,
        "catalog_scaled": catalog_scaled,
        "scaler": scaler,
        "recency_weights": recency_weights,
        "user_profile": user_profile,
    }
