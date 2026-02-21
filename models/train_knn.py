from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as: python models/train_knn.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.feature_engineering import FeatureEngineeringConfig
from models.recommenders.knn import KNNRecommenderConfig, run_knn_recommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run item-to-item KNN music recommender.")
    parser.add_argument("--user-history-csv", default="data/raw/df_all.csv")
    parser.add_argument("--catalog-csv", default="data/raw/spotify-tracks.csv")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--per-track-k", type=int, default=50)
    parser.add_argument("--max-user-tracks", type=int, default=0, help="0 means use all")
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument(
        "--dedupe-mode",
        choices=["track_name_artists", "track_name", "track_id"],
        default="track_name_artists",
    )
    parser.add_argument("--recency-halflife-days", type=float, default=14.0)
    parser.add_argument(
        "--output",
        default="data/processed/recommendations_knn.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--include-debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fe_cfg = FeatureEngineeringConfig(
        user_history_csv=args.user_history_csv,
        catalog_csv=args.catalog_csv,
        exclude_seen_tracks=True,
        recency_halflife_days=args.recency_halflife_days,
    )
    model_cfg = KNNRecommenderConfig(
        top_k=args.top_k,
        per_track_k=args.per_track_k,
        max_user_tracks=None if args.max_user_tracks <= 0 else args.max_user_tracks,
        dedupe_mode=args.dedupe_mode,
        min_similarity=args.min_similarity,
        include_debug=args.include_debug,
    )

    recs, artifacts = run_knn_recommender(fe_config=fe_cfg, model_config=model_cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out, index=False)

    print(f"Wrote {len(recs)} recommendations to {out}")
    print(f"User tracks (cleaned): {artifacts['user_df'].shape[0]}")
    print(f"Catalog candidates: {artifacts['catalog_df'].shape[0]}")
    print(f"Feature columns: {len(artifacts['feature_columns'])}")


if __name__ == "__main__":
    main()

