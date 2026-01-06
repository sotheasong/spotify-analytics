import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

# Base directory where daily snapshots are saved
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "data" / "history"

# Database connection
load_dotenv()

engine = create_engine(f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}")

# Map CSV filenames to table names and columns
DATASETS = {
    "recent_tracks_audio_features.csv": {
        "table": "recent_tracks_audio_features",
        "columns": [
            "id", "name", "acousticness", "danceability", "energy",
            "instrumentalness", "key", "liveness", "mode", "speechiness",
            "valence", "temp"
        ]
    },
    "recent_tracks.csv": {
        "table": "recent_tracks",
        "columns": ["track_id", "track_name", "album_name", "played_at", "artist_name"]
    },
    "top_artists.csv": {
        "table": "top_artists",
        "columns": ["Unnamed: 0", "id", "name", "popularity", "genres", "follower_count"]
    },
    "top_tracks_audio_features.csv": {
        "table": "top_tracks_audio_features",
        "columns": ["Unnamed: 0", "id", "name", "acousticness", "danceability", "energy",
                    "instrumentalness", "key", "liveness", "mode", "speechiness",
                    "valence", "temp"]
    },
    "top_tracks.csv": {
        "table": "top_tracks",
        "columns": ["Unnamed: 0", "id", "track_name", "artist_name", "album_name",
                    "release_date", "duration_min", "popularity", "explicit"]
    }
}

# Process each snapshot folder
for snapshot_dir in sorted(BASE_DIR.iterdir(), key=lambda x: x.stat().st_mtime):
    if not snapshot_dir.is_dir():
        continue

    # Recursively find all CSVs in this snapshot folder
    csv_files = list(snapshot_dir.rglob("*.csv"))
    if not csv_files:
        print(f"Skipping {snapshot_dir.name}: no CSV files found")
        continue

    # Assign collection_date from top-level folder name or fallback
    try:
        collection_date = datetime.strptime(snapshot_dir.name, "%Y-%m-%d").date()
    except ValueError:
        collection_date = datetime.fromtimestamp(snapshot_dir.stat().st_mtime).date()

    print(f"\nIngesting snapshot: {snapshot_dir.name} → collection_date = {collection_date}")

    for file_path in csv_files:
        file_name = file_path.name
        if file_name not in DATASETS:
            print(f"  Skipping unknown file: {file_name}")
            continue

        table_name = DATASETS[file_name]["table"]
        columns = DATASETS[file_name]["columns"]

        # Read CSV
        df = pd.read_csv(file_path)

        # Keep only expected columns, including Unnamed: 0
        df = df[[col for col in columns if col in df.columns]]

        # Add collection_date
        df["collection_date"] = collection_date

        # Insert into SQL
        df.to_sql(
            table_name,
            engine,
            if_exists="append",
            index=False,
            method="multi"
        )

        print(f"  Loaded {file_name} → {table_name}")
