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

from analysis.cleaning import clean_top_artists, clean_recents, clean_top_tracks, clean_audio_features
from analysis.analysis import genre_chart

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

  return redirect("/get-info")


@app.route("/get-info")
def get_info():
    access_token = session.get('access_token')
    if not access_token:
        return redirect("/login")

    if datetime.now().timestamp() > session.get('expires_at', 0):
        return redirect("/refresh_token")

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

    # Collect Spotify IDs
    recent_ids = [t["track"]["id"] for t in recent]
    top_ids = [t["id"] for t in tracks]

    # -----------------------------
    # Helper: Batch a list into n-sized chunks
    # -----------------------------
    def batch(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i+size]

    # -----------------------------
    # Helper: Fetch audio features (Reccobeats)
    # -----------------------------
    def fetch_audio_features(ids):
        url = f"https://api.reccobeats.com/v1/audio-features?ids={','.join(ids)}"
        res = requests.get(url, headers={"Accept": "application/json"})
        return res.json().get("content", [])

    # -----------------------------
    # Helper: Fetch track names (Spotify)
    # -----------------------------
    def fetch_track_metadata(ids):
        url = f"{API_BASE_URL}tracks?ids={','.join(ids)}"
        res = requests.get(url, headers=headers_spotify)
        return res.json().get("tracks", [])

    # ----------------------------------------
    # -------- PROCESS RECENT TRACKS ---------
    # ----------------------------------------
    recent_features = []
    recent_name_meta = []

    # Reccobeats batching: 40 per request
    for chunk in batch(recent_ids, 40):
        feats = fetch_audio_features(chunk)

        # attach Spotify ID to each feature
        for sp_id, feat in zip(chunk, feats):
            feat["spotify_id"] = sp_id

        recent_features.extend(feats)

        # Spotify metadata batching: 50 per request
        recent_name_meta.extend(fetch_track_metadata(chunk))

    df_audio_recent = pd.json_normalize(recent_features)
    df_audio_recent["id"] = df_audio_recent["spotify_id"]

    # map names
    name_lookup_recent = {t["id"]: t["name"] for t in recent_name_meta}
    df_audio_recent["name"] = df_audio_recent["id"].map(name_lookup_recent)

    # ----------------------------------------
    # -------- PROCESS TOP TRACKS ------------
    # ----------------------------------------
    top_features = []
    top_name_meta = []

    for chunk in batch(top_ids, 40):
        feats = fetch_audio_features(chunk)

        for sp_id, feat in zip(chunk, feats):
            feat["spotify_id"] = sp_id

        top_features.extend(feats)
        top_name_meta.extend(fetch_track_metadata(chunk))

    df_audio_top = pd.json_normalize(top_features)
    df_audio_top["id"] = df_audio_top["spotify_id"]

    name_lookup_top = {t["id"]: t["name"] for t in top_name_meta}
    df_audio_top["name"] = df_audio_top["id"].map(name_lookup_top)

    # ----------------------------------------
    # -------- CLEAN + SAVE RESULTS ----------
    # ----------------------------------------
    df_audio_recent = clean_audio_features(df_audio_recent)
    df_audio_top = clean_audio_features(df_audio_top)

    df_audio_recent.to_csv("../data/processed_audio_features_recents.csv", index=False)
    df_tracks.to_csv("../data/processed_tracks.csv", index=True)
    df_artists.to_csv("../data/processed_artists.csv", index=True)
    df_recent.to_csv("../data/processed_recent.csv", index=True)
    df_audio_top.to_csv("../data/processed_audio_features.csv", index=False)

    return jsonify({"status": "data fetched and cleaned"})



@app.route("/refresh_token")
def refresh_token():
  refresh_token = session.get('refresh_token')
  if not refresh_token:
    return redirect("/login")
  
  if datetime.now().timestamp() > session['expires_at']:
    req_body = {
      'grant_type': 'refresh_token',
      'refresh_token': session['refresh_token'],
      'client_id': client_id,
      'client_secret': client_secret
    }
  
    response = requests.post(TOKEN_URL, data=req_body)
    token_info = response.json()
    session['access_token'] = token_info['access_token']
    session['expires_at'] = datetime.now().timestamp() + token_info['expires_in']

    return redirect("/get-info")
  
CORS(app)

if __name__ == "__app__":
  app.run(host="0.0.0.0", debug=True)