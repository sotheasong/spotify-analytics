from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]

# These features match the clustering setup in `notebooks/EDA.ipynb` (cell defining `features = [...]`).
MOOD_FEATURES: tuple[str, ...] = (
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "valence",
    "tempo",
)


DEFAULT_MOOD_K: int = 7

# Legacy labels you used previously for k=7 in EDA writeups.
# LEGACY_MOOD_LABELS_K7: dict[int, str] = {
#     0: "Rhythm-Driven Chill Pop",
#     1: "High-Energy Vocal Tracks",
#     2: "Upbeat Dance Music",
#     3: "High-Intensity Instrumentals",
#     4: "Acoustic & Low-Energy Tracks",
#     5: "Energetic Live-Feel Tracks",
#     6: "High-Tempo Speech-Heavy Tracks",
# }


def default_labels(n_clusters: int) -> dict[int, str]:
    return {i: f"Mood {i}" for i in range(int(n_clusters))}


@dataclass(slots=True)
class MoodModel:
    feature_columns: tuple[str, ...]
    scaler: StandardScaler
    kmeans: KMeans
    labels: dict[int, str]


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def train_mood_model(
    user_history_df: pd.DataFrame,
    *,
    n_clusters: int = DEFAULT_MOOD_K,
    random_state: int = 42,
    feature_columns: tuple[str, ...] = MOOD_FEATURES,
    labels: dict[int, str] | None = None,
) -> MoodModel:
    """
    Train a mood clustering model consistent with EDA:
    - X = df[features].dropna()
    - StandardScaler().fit_transform(X)
    - KMeans(n_clusters=7, random_state=42).fit(X_scaled)
    """
    df = _coerce_numeric(user_history_df, feature_columns)
    X = df[list(feature_columns)].dropna()
    if X.empty:
        raise ValueError("Cannot train mood model: no rows after dropping NA feature values.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X_scaled)

    if labels is None:
        # Use legacy k=7 labels if applicable, otherwise generic.
        labels = LEGACY_MOOD_LABELS_K7.copy() if n_clusters == 7 else default_labels(n_clusters)

    return MoodModel(
        feature_columns=feature_columns,
        scaler=scaler,
        kmeans=kmeans,
        labels=labels,
    )


def save_mood_model(model: MoodModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_mood_model(path: str | Path) -> MoodModel:
    return joblib.load(Path(path))


def ensure_mood_model(
    *,
    user_history_csv: str,
    artifact_path: str | Path | None = None,
    n_clusters: int = DEFAULT_MOOD_K,
    random_state: int = 42,
) -> MoodModel:
    """
    Load a persisted mood model if present; otherwise train from user history and persist it.
    """
    if artifact_path is None:
        artifact_path = REPO_ROOT / "models/artifacts" / f"mood_model_k{int(n_clusters)}.joblib"
    artifact_path = Path(artifact_path)
    if artifact_path.exists():
        return load_mood_model(artifact_path)

    df = pd.read_csv(user_history_csv)
    model = train_mood_model(df, n_clusters=n_clusters, random_state=random_state)
    save_mood_model(model, artifact_path)
    return model


def predict_moods(model: MoodModel, df: pd.DataFrame) -> np.ndarray:
    """
    Predict mood cluster ids for rows in df.
    Rows with missing features get mood_id = -1.
    """
    features = list(model.feature_columns)
    dfn = _coerce_numeric(df, features)
    X = dfn[features]
    valid_mask = ~X.isna().any(axis=1)
    mood_ids = np.full(shape=(len(df),), fill_value=-1, dtype=int)
    if valid_mask.any():
        X_scaled = model.scaler.transform(X.loc[valid_mask])
        mood_ids[valid_mask.to_numpy()] = model.kmeans.predict(X_scaled)
    return mood_ids


def mood_id_to_name(model: MoodModel, mood_id: int) -> str:
    return model.labels.get(int(mood_id), f"Cluster {mood_id}")


def centroids_unscaled(model: MoodModel) -> pd.DataFrame:
    """
    Return cluster centroids in original feature units.
    """
    centers_scaled = pd.DataFrame(model.kmeans.cluster_centers_, columns=list(model.feature_columns))
    centers = pd.DataFrame(model.scaler.inverse_transform(centers_scaled), columns=list(model.feature_columns))
    centers.insert(0, "mood_id", range(centers.shape[0]))
    return centers

