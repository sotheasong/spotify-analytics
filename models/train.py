from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as: python models/train.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.feature_engineering import FeatureEngineeringConfig
from models.recommenders.cosine import CosineRecommenderConfig, run_cosine_recommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cosine music recommender baseline.")
    parser.add_argument(
        "--user-history-csv",
        default="data/raw/df_all.csv",
        help="Path to user listening history CSV.",
    )
    parser.add_argument(
        "--catalog-csv",
        default="data/raw/spotify-tracks.csv",
        help="Path to catalog CSV.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of recommendations to return.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/recommendations_cosine.csv",
        help="Output CSV path for ranked recommendations.",
    )
    parser.add_argument(
        "--include-feature-distance",
        action="store_true",
        help="Include L2 distance to user profile in output.",
    )
    parser.add_argument(
        "--keep-seen",
        action="store_true",
        help="If set, do not exclude tracks already listened to by user.",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=["track_name_artists", "track_name", "track_id"],
        default="track_name_artists",
        help=(
            "How to deduplicate recommendations. "
            "'track_name' removes same song titles across albums/IDs."
        ),
    )
    parser.add_argument(
        "--recency-halflife-days",
        type=float,
        default=14.0,
        help="Half-life (in days) for recency weighting in user profile.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fe_config = FeatureEngineeringConfig(
        user_history_csv=args.user_history_csv,
        catalog_csv=args.catalog_csv,
        exclude_seen_tracks=not args.keep_seen,
        recency_halflife_days=args.recency_halflife_days,
    )
    model_config = CosineRecommenderConfig(
        top_k=args.top_k,
        include_feature_distance=args.include_feature_distance,
        dedupe_mode=args.dedupe_mode,
    )

    recs, artifacts = run_cosine_recommender(fe_config=fe_config, model_config=model_config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recs.to_csv(output_path, index=False)

    print(f"Wrote {len(recs)} recommendations to {output_path}")
    print(f"User tracks (cleaned): {artifacts['user_df'].shape[0]}")
    print(f"Catalog candidates: {artifacts['catalog_df'].shape[0]}")
    print(f"Feature columns: {len(artifacts['feature_columns'])}")


if __name__ == "__main__":
    main()
