from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from models.feature_engineering import FeatureEngineeringConfig, run_feature_engineering
from models.recommenders.genre_similarity import (
    blended_rank_score,
    build_genre_similarity_artifacts,
    cosine_similarity_to_profile,
    rank_normalize,
)


@dataclass(slots=True)
class CosineRecommenderConfig:
    top_k: int = 50
    include_feature_distance: bool = False
    deduplicate_tracks: bool = True
    # Options: "track_name_artists", "track_name", "track_id"
    dedupe_mode: str = "track_name_artists"
    # Blend in genre similarity (0 = disabled, 1 = only genre).
    genre_weight: float = 0.0
    # Blend in catalog popularity as a re-rank term (0 = disabled, 1 = only popularity).
    popularity_weight: float = 0.0


def cosine_similarity_scores(user_profile: np.ndarray, catalog_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one user vector and all catalog vectors.
    """
    user_vec = np.asarray(user_profile, dtype=float).reshape(-1)
    catalog = np.asarray(catalog_matrix, dtype=float)
    if catalog.ndim != 2:
        raise ValueError("catalog_matrix must be 2D.")
    if user_vec.shape[0] != catalog.shape[1]:
        raise ValueError("user_profile dimension must match catalog feature dimension.")

    user_norm = np.linalg.norm(user_vec)
    if user_norm <= 0:
        return np.zeros(catalog.shape[0], dtype=float)

    catalog_norms = np.linalg.norm(catalog, axis=1)
    denom = np.clip(catalog_norms * user_norm, a_min=1e-12, a_max=None)
    numer = catalog @ user_vec
    return numer / denom


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().casefold()
    # Collapse repeated whitespace.
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
    config: CosineRecommenderConfig | None = None,
) -> pd.DataFrame:
    """
    Rank catalog tracks by cosine similarity to the user profile.
    """
    cfg = config or CosineRecommenderConfig()
    if cfg.top_k <= 0:
        raise ValueError("top_k must be > 0.")
    if cfg.dedupe_mode not in {"track_name_artists", "track_name", "track_id"}:
        raise ValueError("dedupe_mode must be one of: track_name_artists, track_name, track_id")

    user_df: pd.DataFrame = artifacts["user_df"]
    catalog_df: pd.DataFrame = artifacts["catalog_df"]
    catalog_id_col: str = artifacts["catalog_id_column"]
    catalog_scaled: np.ndarray = artifacts["catalog_scaled"]
    user_profile: np.ndarray = artifacts["user_profile"]

    audio_scores = cosine_similarity_scores(user_profile=user_profile, catalog_matrix=catalog_scaled)

    # Optional genre similarity (rank-normalized blend).
    genre_scores: np.ndarray | None = None
    final_scores = audio_scores
    if cfg.genre_weight and float(cfg.genre_weight) > 0.0:
        recency = artifacts.get("recency_weights", None)
        genre_art = build_genre_similarity_artifacts(
            user_df=user_df,
            catalog_df=catalog_df,
            recency_weights=recency,
        )
        if genre_art is not None:
            genre_scores = cosine_similarity_to_profile(genre_art.catalog_matrix, genre_art.user_profile)
            final_scores = blended_rank_score(
                base_scores=audio_scores,
                genre_scores=genre_scores,
                genre_weight=cfg.genre_weight,
            )

    # Optional popularity re-rank (rank-normalized blend).
    popularity_norm: np.ndarray | None = None
    if cfg.popularity_weight and float(cfg.popularity_weight) > 0.0 and "popularity" in catalog_df.columns:
        popularity_raw = pd.to_numeric(catalog_df["popularity"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pop_norm = rank_normalize(popularity_raw)
        popularity_norm = pop_norm
        base_norm = rank_normalize(final_scores)
        w = float(cfg.popularity_weight)
        w = min(max(w, 0.0), 1.0)
        final_scores = (1.0 - w) * base_norm + w * pop_norm

    top_k = min(cfg.top_k, len(final_scores))
    if top_k == 0:
        return pd.DataFrame(columns=[catalog_id_col, "score", "rank"])

    sorted_idx = np.argsort(final_scores)[::-1]

    if cfg.deduplicate_tracks:
        selected: list[int] = []
        seen_keys: set[str] = set()
        for idx in sorted_idx:
            key = _dedupe_key(
                catalog_df.iloc[idx],
                catalog_id_col=catalog_id_col,
                dedupe_mode=cfg.dedupe_mode,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(int(idx))
            if len(selected) >= top_k:
                break
        top_idx = np.asarray(selected, dtype=int)
    else:
        top_idx = sorted_idx[:top_k]

    # Keep useful metadata columns when present.
    preferred_cols = [
        catalog_id_col,
        "track_name",
        "artists",
        "album_name",
        "track_genre",
    ]
    keep_cols = [c for c in preferred_cols if c in catalog_df.columns]

    result = catalog_df.iloc[top_idx][keep_cols].copy()
    result["score"] = final_scores[top_idx]
    result["audio_score"] = audio_scores[top_idx]
    if genre_scores is not None:
        result["genre_score"] = genre_scores[top_idx]
    if popularity_norm is not None:
        result["popularity_norm"] = popularity_norm[top_idx]
    result["rank"] = np.arange(1, len(result) + 1, dtype=int)

    if cfg.include_feature_distance and "feature_columns" in artifacts:
        # L2 distance is useful for debugging and secondary sorting.
        user_vec = np.asarray(user_profile, dtype=float).reshape(1, -1)
        diff = catalog_scaled[top_idx] - user_vec
        result["l2_distance"] = np.linalg.norm(diff, axis=1)

    return result.reset_index(drop=True)


def run_cosine_recommender(
    fe_config: FeatureEngineeringConfig | None = None,
    model_config: CosineRecommenderConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convenience wrapper:
    1) run feature engineering
    2) return top-k cosine recommendations + artifacts
    """
    artifacts = run_feature_engineering(fe_config)
    recs = recommend_from_artifacts(artifacts=artifacts, config=model_config)
    return recs, artifacts

