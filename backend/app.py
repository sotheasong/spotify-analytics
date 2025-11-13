import json
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request, session
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor


from analysis.cleaning import (
    clean_recents,
    clean_top_artists,
    clean_top_tracks,
    clean_audio_features
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

REDIRECT_URI = "http://127.0.0.1:5000/callback"

FRONTEND_URI = "http://localhost:5173"

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE_URL = "https://api.spotify.com/v1/"


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DATA_DIR = REPO_ROOT / "data"
HISTORY_DIR = DATA_DIR / "history"
AUTH_DIR = DATA_DIR / "auth"
REFRESH_TOKEN_PATH = AUTH_DIR / "refresh_token.json"
PROCESSED_TRACKS_PATH = DATA_DIR / "processed_tracks.csv"
PROCESSED_ARTISTS_PATH = DATA_DIR / "processed_artists.csv"
PROCESSED_RECENT_PATH = DATA_DIR / "processed_recent.csv"
PROCESSED_TOP_AUDIO_FEATURES_PATH = DATA_DIR / "processed_top_track_audio_features.csv"
PROCESSED_RECENT_AUDIO_FEATURES_PATH = DATA_DIR / "processed_recent_track_audio_features.csv"


def ensure_directories() -> None:
  """Ensure that filesystem locations used by the app exist."""
  for path in (DATA_DIR, HISTORY_DIR, AUTH_DIR):
    path.mkdir(parents=True, exist_ok=True)


ensure_directories()


def store_refresh_token(refresh_token_value: str) -> None:
  """Persist the refresh token for use in offline jobs."""
  if not refresh_token_value:
    return

  AUTH_DIR.mkdir(parents=True, exist_ok=True)
  REFRESH_TOKEN_PATH.write_text(json.dumps({"refresh_token": refresh_token_value}))


def load_persisted_refresh_token() -> Optional[str]:
  """Read a refresh token persisted by ``store_refresh_token``."""
  if not REFRESH_TOKEN_PATH.exists():
    return None

  try:
    data = json.loads(REFRESH_TOKEN_PATH.read_text())
  except json.JSONDecodeError:
    return None

  return data.get("refresh_token")


def refresh_access_token(refresh_token_value: str) -> dict:
  """Exchange a refresh token for a new Spotify access token."""
  req_body = {
      'grant_type': 'refresh_token',
      'refresh_token': refresh_token_value,
      'client_id': client_id,
      'client_secret': client_secret
  }

  response = requests.post(TOKEN_URL, data=req_body)
  response.raise_for_status()
  token_info = response.json()

  new_refresh_token = token_info.get('refresh_token')
  if new_refresh_token:
    store_refresh_token(new_refresh_token)

  return token_info


def collect_user_datasets(access_token: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  headers_spotify = {"Authorization": f"Bearer {access_token}"}

  # Fetch main data
  tracks = requests.get(
      f"{API_BASE_URL}me/top/tracks?time_range=long_term&limit=50",
      headers=headers_spotify
  ).json()['items']

  artists = requests.get(
      f"{API_BASE_URL}me/top/artists?time_range=long_term&limit=50",
      headers=headers_spotify
  ).json()['items']

  recent = requests.get(
      f"{API_BASE_URL}me/player/recently-played?limit=50",
      headers=headers_spotify
  ).json()['items']

  df_tracks = clean_top_tracks(tracks)
  df_artists = clean_top_artists(artists)
  df_recent = clean_recents(recent)

  # Spotify IDs
  recent_ids = [t["track"]["id"] for t in recent]
  top_ids = [t["id"] for t in tracks]

  # Batch helper
  def batch(lst, size):
      for i in range(0, len(lst), size):
          yield lst[i:i+size]

  # Fetch Reccobeats features
  def fetch_audio_features(ids):
      url = f"https://api.reccobeats.com/v1/audio-features?ids={','.join(ids)}"
      res = requests.get(url, headers={"Accept": "application/json"})
      return res.json().get("content", [])

  # Fetch Spotify names
  def fetch_track_metadata(ids):
      url = f"{API_BASE_URL}tracks?ids={','.join(ids)}"
      res = requests.get(url, headers=headers_spotify)
      return res.json().get("tracks", [])

  # ----------------------------------------------------
  # ------------ HELPER: Safe Matching Logic -----------
  # ----------------------------------------------------
  def safe_match_features(spotify_ids):
      all_features = []
      all_names = []

      for chunk in batch(spotify_ids, 40):
          returned_feats = fetch_audio_features(chunk)
          returned_names = fetch_track_metadata(chunk)

          # returned_feats may be shorter than chunk!
          returned_feats_map = {
              idx: returned_feats[idx]
              for idx in range(len(returned_feats))
          }

          # Process each Spotify ID in the original order
          for i, sp_id in enumerate(chunk):

              if i in returned_feats_map:
                  feat = returned_feats_map[i]
                  feat["spotify_id"] = sp_id
                  feat["missing_audio_features"] = False
                  all_features.append(feat)
              else:
                  # Missing audio features → create placeholder row
                  all_features.append({
                      "spotify_id": sp_id,
                      "missing_audio_features": True
                  })

          all_names.extend(returned_names)

      return all_features, all_names

  # ----------------------------------------------------
  # Process RECENT (50 rows GUARANTEED)
  # ----------------------------------------------------
  recent_features, recent_name_meta = safe_match_features(recent_ids)

  df_audio_recent = pd.json_normalize(recent_features)
  df_audio_recent["id"] = df_audio_recent["spotify_id"]

  name_lookup_recent = {t["id"]: t["name"] for t in recent_name_meta}
  df_audio_recent["name"] = df_audio_recent["id"].map(name_lookup_recent)

  # ----------------------------------------------------
  # Process TOP TRACKS (50 rows GUARANTEED)
  # ----------------------------------------------------
  top_features, top_name_meta = safe_match_features(top_ids)

  df_audio_top = pd.json_normalize(top_features)
  df_audio_top["id"] = df_audio_top["spotify_id"]

  name_lookup_top = {t["id"]: t["name"] for t in top_name_meta}
  df_audio_top["name"] = df_audio_top["id"].map(name_lookup_top)

  # ----------------------------------------------------
  # Clean and save
  # ----------------------------------------------------
  df_audio_recent = clean_audio_features(df_audio_recent)
  df_audio_top = clean_audio_features(df_audio_top)
  return df_tracks, df_artists, df_recent, df_audio_top, df_audio_recent

def persist_snapshot(
    df_tracks: pd.DataFrame,
    df_artists: pd.DataFrame,
    df_recent: pd.DataFrame,
    df_audio_recent: pd.DataFrame = None,
    df_audio_top: pd.DataFrame = None,
    snapshot_time: Optional[datetime] = None
) -> Path:
  """Persist the cleaned datasets to the processed files and timestamped history."""
  snapshot_time = snapshot_time or datetime.now(timezone.utc)
  date_dir = HISTORY_DIR / snapshot_time.strftime("%Y-%m-%d")
  snapshot_dir = date_dir / snapshot_time.strftime("%H%M%S")
  snapshot_dir.mkdir(parents=True, exist_ok=True)

  df_tracks.to_csv(snapshot_dir / "top_tracks.csv", index=True)
  df_artists.to_csv(snapshot_dir / "top_artists.csv", index=True)
  df_recent.to_csv(snapshot_dir / "recent_tracks.csv", index=False)
  df_audio_recent.to_csv(snapshot_dir / "recent_tracks_audio_features.csv", index=False)
  df_audio_top.to_csv(snapshot_dir / "top_tracks_audio_features.csv", index=True)

  df_tracks.to_csv(PROCESSED_TRACKS_PATH, index=True)
  df_artists.to_csv(PROCESSED_ARTISTS_PATH, index=True)
  df_recent.to_csv(PROCESSED_RECENT_PATH, index=False)
  df_audio_recent.to_csv(PROCESSED_RECENT_AUDIO_FEATURES_PATH, index=False)
  df_audio_top.to_csv(PROCESSED_TOP_AUDIO_FEATURES_PATH, index=True)
  return snapshot_dir

@app.route("/")
def index():
  return "Welcome to the Spotify Analytics App! <a href='/login'>Login with Spotify</a>"

@app.route("/login")
def login():
  scope = "user-read-private user-read-email user-read-playback-position user-top-read user-read-recently-played"
  params = {
      "response_type": "code",
      "redirect_uri": REDIRECT_URI,
      "scope": scope,
      "client_id": client_id,
      "show_dialog": True
  }
  
  auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"
  
  return redirect(auth_url)


@app.route("/callback")
def callback():
  if 'error' in request.args:
    return jsonify({"error": request.args['error']})
  
  if 'code' in request.args:
    req_body = {
      'code': request.args['code'],
      'grant_type': 'authorization_code',
      'redirect_uri': REDIRECT_URI,
      'client_id': client_id,
      'client_secret': client_secret
    }

  response = requests.post(TOKEN_URL, data=req_body)
  token_info = response.json()

  session['access_token'] = token_info['access_token']
  session['refresh_token'] = token_info['refresh_token']
  session['expires_at'] = token_info['expires_in'] 
  session['expires_at'] = datetime.now().timestamp() + token_info['expires_in']
  store_refresh_token(token_info.get('refresh_token'))

  return redirect("/get-info")


@app.route("/get-info")
def get_info():
    access_token = session.get('access_token')
    if not access_token:
        return redirect("/login")

    if datetime.now().timestamp() > session.get('expires_at', 0):
        return redirect("/refresh_token")

    df_tracks, df_artists, df_recent, df_audio_top, df_audio_recent = collect_user_datasets(access_token)
    persist_snapshot(df_tracks, df_artists, df_recent, df_audio_recent, df_audio_top)

    return jsonify({"status": "data fetched and cleaned"})



@app.route("/refresh_token")
def refresh_token():
  refresh_token = session.get('refresh_token')
  if not refresh_token:
    return redirect("/login")

  if datetime.now().timestamp() > session['expires_at']:
    token_info = refresh_access_token(refresh_token)
    session['access_token'] = token_info['access_token']
    session['expires_at'] = datetime.now().timestamp() + token_info['expires_in']
    session['refresh_token'] = token_info.get('refresh_token', refresh_token)

    return redirect("/get-info")
  
CORS(app)

if __name__ == "__main__":
  app.run(host="0.0.0.0", debug=True)