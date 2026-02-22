from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    # Treat NaN as empty.
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
    except Exception:
        pass
    return str(value)


def _basic_normalize(token: str) -> str:
    t = token.strip().casefold()
    # Keep it simple: collapse whitespace.
    t = " ".join(t.split())
    return t


def _candidate_variants(token: str) -> list[str]:
    """
    Generate common variants to match the catalog taxonomy.
    """
    t = _basic_normalize(token)
    out = [t]
    if " " in t:
        out.append(t.replace(" ", "-"))
        out.append(t.replace(" ", "-").replace("&", "and"))
    if "&" in t:
        out.append(t.replace("&", "and"))
    return list(dict.fromkeys(out))


def _canonicalize_to_catalog(token: str, *, catalog_allowed: set[str]) -> str | None:
    """
    Map a raw Spotify artist-genre token to the catalog's coarse `track_genre` taxonomy.

    Rule priority:
    - exact / hyphenated match in catalog
    - strip common region prefixes and retry
    - hand-written synonym map for known mismatches
    - keyword bucket rules (e.g., all rap/hip-hop subgenres -> hip-hop)
    """
    raw = _basic_normalize(token)
    if not raw or raw == "nan":
        return None

    # Direct match attempts.
    for cand in _candidate_variants(raw):
        if cand in catalog_allowed:
            return cand

    # Strip common prefixes (country/region/style qualifiers) and retry.
    prefixes = (
        "uk ",
        "us ",
        "french ",
        "german ",
        "japanese ",
        "turkish ",
        "vietnamese ",
        "argentine ",
        "chilean ",
        "brazilian ",
        "mexican ",
        "spanish ",
        "latin ",
        "east coast ",
        "west coast ",
        "southern ",
        "new york ",
        "brooklyn ",
    )
    stripped = raw
    for p in prefixes:
        if stripped.startswith(p):
            stripped = stripped[len(p) :]
            break
    if stripped != raw:
        for cand in _candidate_variants(stripped):
            if cand in catalog_allowed:
                return cand

    # Explicit synonyms / known mismatches.
    synonym_map = {
        "hip hop": "hip-hop",
        "hiphop": "hip-hop",
        "rap": "hip-hop",
        "drum and bass": "drum-and-bass",
        "dnb": "drum-and-bass",
        "r&b": "r-n-b",
        "rnb": "r-n-b",
        "r and b": "r-n-b",
        "alt rock": "alt-rock",
        "alternative rock": "alt-rock",
        "punk rock": "punk-rock",
        "rock n roll": "rock-n-roll",
        "rock and roll": "rock-n-roll",
        "singer songwriter": "singer-songwriter",
        "show tunes": "show-tunes",
        "world music": "world-music",
    }
    mapped = synonym_map.get(raw) or synonym_map.get(stripped)
    if mapped and mapped in catalog_allowed:
        return mapped

    # Keyword buckets to group related subgenres into the catalog taxonomy.
    def has_any(substrs: tuple[str, ...]) -> bool:
        return any(s in raw for s in substrs)

    if "hip-hop" in catalog_allowed and has_any(
        (
            "hip hop",
            "hip-hop",
            "rap",
            "boom bap",
            "bap",
            "drill",
            "grime",
            "trap",
            "phonk",
            "crunk",
            "g-funk",
            "g funk",
            "mumble rap",
            "emo rap",
            "cloud rap",
            "horrorcore",
        )
    ):
        return "hip-hop"

    if "r-n-b" in catalog_allowed and has_any(("r&b", "rnb", "r and b", "neo soul", "soul")):
        # Prefer r-n-b for r&b/neo soul if available.
        return "r-n-b"

    # Broad buckets for common electronic subgenres.
    if "techno" in catalog_allowed and "techno" in raw:
        return "techno"
    if "house" in catalog_allowed and "house" in raw:
        return "house"
    if "trance" in catalog_allowed and "trance" in raw:
        return "trance"
    if "dubstep" in catalog_allowed and "dubstep" in raw:
        return "dubstep"
    if "drum-and-bass" in catalog_allowed and ("drum" in raw and "bass" in raw):
        return "drum-and-bass"

    # Broad buckets for rock/pop/metal/jazz/reggae/etc.
    if "pop" in catalog_allowed and "pop" in raw:
        return "pop"
    if "rock" in catalog_allowed and "rock" in raw:
        return "rock"
    if "metal" in catalog_allowed and "metal" in raw:
        return "metal"
    if "jazz" in catalog_allowed and "jazz" in raw:
        return "jazz"
    if "reggae" in catalog_allowed and "reggae" in raw:
        return "reggae"
    if "classical" in catalog_allowed and "classical" in raw:
        return "classical"
    if "ambient" in catalog_allowed and "ambient" in raw:
        return "ambient"

    return None


def _split_genres(text: Any, *, catalog_allowed: set[str] | None = None) -> list[str]:
    """
    Split a comma-separated genre string into normalized labels.

    If catalog_allowed is provided, tokens are canonicalized into the catalog taxonomy.
    """
    raw = _as_text(text).strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]

    if catalog_allowed is None:
        return [_basic_normalize(p) for p in parts if _basic_normalize(p)]

    out: list[str] = []
    for p in parts:
        canon = _canonicalize_to_catalog(p, catalog_allowed=catalog_allowed)
        if canon:
            out.append(canon)
    # unique, stable
    return list(dict.fromkeys(out))


def resolve_genre_text_column(df: pd.DataFrame) -> str | None:
    """
    Best-effort column resolver for genre-like text.
    """
    for col in ("artist_genres", "genres", "track_genre"):
        if col in df.columns:
            return col
    return None


@dataclass(slots=True)
class GenreSimilarityArtifacts:
    genre_column_user: str
    genre_column_catalog: str
    mlb: MultiLabelBinarizer
    user_profile: np.ndarray  # dense, shape (n_genres,)
    catalog_matrix: Any  # sparse matrix


def build_genre_similarity_artifacts(
    *,
    user_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    recency_weights: np.ndarray | None = None,
    user_genre_col: str | None = None,
    catalog_genre_col: str | None = None,
) -> GenreSimilarityArtifacts | None:
    """
    Build reusable genre representations for user + catalog.

    Returns None if no suitable genre columns exist.
    """
    ugc = user_genre_col or resolve_genre_text_column(user_df)
    cgc = catalog_genre_col or resolve_genre_text_column(catalog_df)
    if not ugc or not cgc:
        return None

    # Normalize catalog genre labels to a canonical set we will map into.
    # (spotify-tracks.csv uses a coarse taxonomy like "hip-hop", "r-n-b", "drum-and-bass", etc.)
    catalog_series = catalog_df[cgc].dropna().astype(str).apply(_basic_normalize)
    catalog_allowed = set(catalog_series.unique().tolist())

    user_lists = user_df[ugc].apply(lambda s: _split_genres(s, catalog_allowed=catalog_allowed)).tolist()
    catalog_lists = catalog_df[cgc].apply(lambda s: _split_genres(s, catalog_allowed=catalog_allowed)).tolist()

    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(user_lists + catalog_lists)

    X_user = mlb.transform(user_lists)
    X_cat = mlb.transform(catalog_lists)

    if recency_weights is None:
        w = np.ones(X_user.shape[0], dtype=float)
    else:
        w = np.asarray(recency_weights, dtype=float)
        if w.shape[0] != X_user.shape[0]:
            raise ValueError("recency_weights length does not match user_df rows.")

    # Weighted average of user rows -> dense profile vector.
    w = np.clip(w, a_min=0.0, a_max=None)
    denom = float(w.sum())
    if denom <= 0.0:
        user_profile = np.asarray(X_user.mean(axis=0)).reshape(-1)
    else:
        user_profile = np.asarray((X_user.multiply(w.reshape(-1, 1))).sum(axis=0) / denom).reshape(-1)

    return GenreSimilarityArtifacts(
        genre_column_user=ugc,
        genre_column_catalog=cgc,
        mlb=mlb,
        user_profile=user_profile.astype(float, copy=False),
        catalog_matrix=X_cat,
    )


def cosine_similarity_to_profile(catalog_matrix: Any, user_profile: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between each catalog row (sparse) and user profile (dense).
    """
    prof = np.asarray(user_profile, dtype=float).reshape(-1)
    prof_norm = float(np.linalg.norm(prof))
    if prof_norm <= 0:
        return np.zeros(catalog_matrix.shape[0], dtype=float)

    # numerator: (n_items,)
    numer = np.asarray(catalog_matrix.dot(prof)).reshape(-1)

    # row norms: (n_items,)
    row_sq = np.asarray(catalog_matrix.multiply(catalog_matrix).sum(axis=1)).reshape(-1)
    row_norm = np.sqrt(np.clip(row_sq, a_min=0.0, a_max=None))

    denom = np.clip(row_norm * prof_norm, a_min=1e-12, a_max=None)
    return numer / denom


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores to [0, 1] via rank percentile.

    Ties get an arbitrary but stable ordering; good enough for blending.
    """
    x = np.asarray(scores, dtype=float)
    n = x.shape[0]
    if n == 0:
        return x
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    if n == 1:
        return np.zeros_like(ranks)
    return ranks / (n - 1.0)


def blended_rank_score(
    *,
    base_scores: np.ndarray,
    genre_scores: np.ndarray,
    genre_weight: float,
) -> np.ndarray:
    w = float(genre_weight)
    w = min(max(w, 0.0), 1.0)
    b = rank_normalize(base_scores)
    g = rank_normalize(genre_scores)
    return (1.0 - w) * b + w * g

