from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from models.feature_engineering import FeatureEngineeringConfig, run_feature_engineering


@dataclass(slots=True)
class KNNRecommenderConfig:
    # Final number of recommendations returned.
    top_k: int = 50
    # Number of neighbors to retrieve per listened track.
    per_track_k: int = 50
    # Optional cap on number of user-history tracks to use (most recent first).
    # Set to None to use all user tracks.
    max_user_tracks: int | None = None
    # Dedupe controls (same semantics as cosine recommender).
    deduplicate_tracks: bool = True
    # Options: "track_name_artists", "track_name", "track_id"
    dedupe_mode: str = "track_name_artists"
    # Drop candidate occurrences below this similarity.
    min_similarity: float = 0.0
    # Add some debugging columns.
    include_debug: bool = False


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().casefold()
    return " ".join(text.split())


def _dedupe_key(row: pd.Series, catalog_id_col: str, dedupe_mode: str) -> str:
    track_name = _normalize_text(row.get("track_name", ""))
    artists = _normalize_text(row.get("artists", ""))
    track_id = _normalize_text(row.get(catalog_id_col, ""))

    if dedupe_mode == "track_name":
        return track_name or track_id
    if dedupe_mode == "track_id":
        return track_id or track_name

    # default: track_name_artists
    if track_name and artists:
        return f"{track_name}||{artists}"
    return track_name or track_id


def recommend_from_artifacts(
    artifacts: dict[str, Any],
    config: KNNRecommenderConfig | None = None,
) -> pd.DataFrame:
    """
    Item-to-item KNN recommender (content-based).

    Steps:
    - For each user track vector, retrieve nearest neighbors from catalog (cosine).
    - Aggregate candidate scores: sum(recency_weight * similarity) across all source tracks.
    - Rank by aggregated score and optionally dedupe.
    """
    cfg = config or KNNRecommenderConfig()
    if cfg.top_k <= 0:
        raise ValueError("top_k must be > 0.")
    if cfg.per_track_k <= 0:
        raise ValueError("per_track_k must be > 0.")
    if cfg.dedupe_mode not in {"track_name_artists", "track_name", "track_id"}:
        raise ValueError("dedupe_mode must be one of: track_name_artists, track_name, track_id")

    user_df: pd.DataFrame = artifacts["user_df"]
    catalog_df: pd.DataFrame = artifacts["catalog_df"]
    catalog_id_col: str = artifacts["catalog_id_column"]
    user_scaled: np.ndarray = artifacts["user_scaled"]
    catalog_scaled: np.ndarray = artifacts["catalog_scaled"]
    recency_weights: np.ndarray = artifacts.get("recency_weights", np.ones(user_scaled.shape[0], dtype=float))

    if user_scaled.size == 0 or catalog_scaled.size == 0:
        return pd.DataFrame(columns=[catalog_id_col, "score", "rank"])

    # Optionally keep only the most recent user tracks.
    if cfg.max_user_tracks is not None and cfg.max_user_tracks > 0 and cfg.max_user_tracks < user_scaled.shape[0]:
        # Prefer using date ordering if available; fall back to end of dataframe.
        if "collection_date" in user_df.columns:
            # IMPORTANT: use positional indices, not DataFrame index labels.
            dt = pd.to_datetime(user_df["collection_date"], errors="coerce")
            dt_filled = dt.fillna(pd.Timestamp.min)
            keep_idx = np.argsort(dt_filled.to_numpy())[::-1][: cfg.max_user_tracks]
        else:
            keep_idx = np.arange(user_scaled.shape[0] - cfg.max_user_tracks, user_scaled.shape[0])

        user_scaled_used = user_scaled[keep_idx]
        recency_used = recency_weights[keep_idx]
    else:
        user_scaled_used = user_scaled
        recency_used = recency_weights

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(catalog_scaled)

    n_neighbors = min(cfg.per_track_k, catalog_scaled.shape[0])
    distances, indices = nn.kneighbors(user_scaled_used, n_neighbors=n_neighbors, return_distance=True)
    similarities = 1.0 - distances

    # Aggregate scores across all user source tracks.
    scores = np.zeros(catalog_scaled.shape[0], dtype=float)
    support = np.zeros(catalog_scaled.shape[0], dtype=int)
    max_sim = np.zeros(catalog_scaled.shape[0], dtype=float)

    for i in range(similarities.shape[0]):
        w = float(recency_used[i]) if i < recency_used.shape[0] else 1.0
        for j in range(similarities.shape[1]):
            sim = float(similarities[i, j])
            if sim < cfg.min_similarity:
                continue
            idx = int(indices[i, j])
            scores[idx] += w * sim
            support[idx] += 1
            if sim > max_sim[idx]:
                max_sim[idx] = sim

    if not np.any(scores):
        return pd.DataFrame(columns=[catalog_id_col, "score", "rank"])

    sorted_idx = np.argsort(scores)[::-1]

    selected: list[int] = []
    if cfg.deduplicate_tracks:
        seen: set[str] = set()
        for idx in sorted_idx:
            if scores[idx] <= 0:
                break
            key = _dedupe_key(catalog_df.iloc[idx], catalog_id_col=catalog_id_col, dedupe_mode=cfg.dedupe_mode)
            if key in seen:
                continue
            seen.add(key)
            selected.append(int(idx))
            if len(selected) >= cfg.top_k:
                break
    else:
        selected = [int(i) for i in sorted_idx[: cfg.top_k] if scores[int(i)] > 0]

    if not selected:
        return pd.DataFrame(columns=[catalog_id_col, "score", "rank"])

    top_idx = np.asarray(selected, dtype=int)

    preferred_cols = [
        catalog_id_col,
        "track_name",
        "artists",
        "album_name",
        "track_genre",
        "popularity",
    ]
    keep_cols = [c for c in preferred_cols if c in catalog_df.columns]

    result = catalog_df.iloc[top_idx][keep_cols].copy()
    result["score"] = scores[top_idx]
    result["rank"] = np.arange(1, len(result) + 1, dtype=int)

    if cfg.include_debug:
        result["support"] = support[top_idx]
        result["max_similarity"] = max_sim[top_idx]

    return result.reset_index(drop=True)


def run_knn_recommender(
    fe_config: FeatureEngineeringConfig | None = None,
    model_config: KNNRecommenderConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    artifacts = run_feature_engineering(fe_config)
    recs = recommend_from_artifacts(artifacts=artifacts, config=model_config)
    return recs, artifacts

