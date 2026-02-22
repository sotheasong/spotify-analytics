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
- Recommender models:
  - `models/recommenders/cosine.py` (cosine baseline)
  - `models/recommenders/knn.py` (item-to-item KNN)
- CLI runners:
  - `models/train.py` (cosine)
  - `models/train_knn.py` (KNN)
- Thin Spotify playlist integration:
  - Backend: `POST /api/recommendations/create-playlist` (supports `model=cosine|knn`)
  - Frontend: `frontend/src/pages/Analytics.jsx` (model + params selector, “Open in Spotify”)

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

### 3) Model 2: Item-to-item KNN (`models/recommenders/knn.py`)

- For each listened track, finds `per_track_k` nearest catalog neighbors (cosine distance in feature space)
- Aggregates candidate scores across all listened tracks using recency weights
- Supports `max_user_tracks` (most recent N source tracks) and `min_similarity` filtering
- Uses the same dedupe modes as cosine (`track_name`, `track_name_artists`, `track_id`)

### 4) CLI runners

#### Cosine runner (`models/train.py`)

- Runs feature engineering + cosine model end-to-end
- Writes recommendations to CSV

Default output:

- `data/processed/recommendations_cosine.csv`

#### KNN runner (`models/train_knn.py`)

- Runs feature engineering + KNN model end-to-end
- Writes recommendations to CSV

Default output:

- `data/processed/recommendations_knn.csv`

## How to run the recommenders

From repo root:

```bash
python models/train.py
```

KNN example:

```bash
python models/train_knn.py \
  --top-k 100 \
  --per-track-k 50 \
  --dedupe-mode track_name \
  --output data/processed/recommendations_knn.csv
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
python app.py
```

Notes:

- Spotify redirect + frontend session works best when you consistently use `127.0.0.1` for both frontend and backend.
- Spotify playlist creation requires playlist scopes (already included in `/login`).

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
