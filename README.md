# spotify-analytics

Personal Spotify analytics + recommendation project.

## Current status

Implemented so far:

- Ingestion and historical tracking into PostgreSQL (`recent_tracks`, `recent_tracks_audio_features`, etc.)
- EDA notebooks:
  - `notebooks/EDA.ipynb` (user listening history)
  - `notebooks/data_EDA.ipynb` (catalog dataset)
- Feature engineering pipeline:
  - `models/feature_engineering.py`
- First recommender model (baseline):
  - `models/recommenders/cosine.py`
- Simple training/runner CLI:
  - `models/train.py`

## Recommender pipeline (implemented)

### 1) Feature engineering (`models/feature_engineering.py`)

The pipeline currently:

- Loads user history from `data/raw/df_all.csv`
- Loads catalog from `data/raw/spotify-tracks.csv`
- Resolves shared numeric feature columns
- Cleans/coerces numeric audio features
- Handles missing values (drop or median fill)
- Deduplicates tracks
- Excludes already-listened tracks (configurable)
- Standardizes features with `StandardScaler`
- Computes recency weights from `collection_date`
- Builds a user preference vector (weighted profile)

### 2) Model 1: Cosine similarity baseline (`models/recommenders/cosine.py`)

- Scores each catalog track against the user profile with cosine similarity
- Ranks tracks by score descending
- Returns top-k recommendations with metadata (`track_name`, `artists`, `album_name`, etc.)

### 3) CLI runner (`models/train.py`)

- Runs feature engineering + cosine model end-to-end
- Writes recommendations to CSV

Default output:

- `data/processed/recommendations_cosine.csv`

## How to run the recommender

From repo root:

```bash
python models/train.py
```

Example with options:

```bash
python models/train.py \
  --top-k 100 \
  --output data/processed/recommendations_cosine.csv \
  --recency-halflife-days 14
```

Optional flags:

- `--include-feature-distance` include L2 distance in output
- `--keep-seen` keep already-listened tracks in candidate pool

## Setup

- Create `.env` file with the following
```bash
DB_USER = 
DB_PASS = 
DB_HOST = 
DB_NAME = 

CLIENT_ID = 
CLIENT_SECRET = 
```

## Backend setup (Flask)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: .\\venv\\Scripts\\activate
pip install -r ../requirements.txt
flask run
```

Alternative:

```bash
python -m backend.app
python -m backend.jobs.daily_snapshot
```

## Frontend setup (Vite + React)

```bash
cd frontend
npm install
npm run dev
```

## Next planned steps

- Add model comparison (weighted centroid + KNN baseline)
- Add offline evaluation script (Hit@K / Recall@K with temporal split)
- Build playlist blending logic (new recommendations + familiar tracks)
