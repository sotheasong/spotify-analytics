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
  
  headers = {
    "Authorization": f"Bearer {access_token}"
  }
  
  tracks_res = requests.get(f"{API_BASE_URL}me/top/tracks?time_range=long_term&limit=50", headers=headers)
  tracks = tracks_res.json()['items']

  artists_res = requests.get(f"{API_BASE_URL}me/top/artists?time_range=long_term&limit=50", headers=headers)
  artists = artists_res.json()['items']

  recent_res = requests.get(f"{API_BASE_URL}me/player/recently-played?limit=50", headers=headers)
  recent = recent_res.json()['items']

  df_tracks = clean_top_tracks(tracks)
  df_artists = clean_top_artists(artists)
  df_recent = clean_recents(recent)

  df_tracks.to_csv("../data/processed_tracks.csv", index=True)
  df_artists.to_csv("../data/processed_artists.csv", index=True)
  df_recent.to_csv("../data/processed_recent.csv", index=True)

  df = pd.DataFrame()
  for t in tracks:
    id = t['id']
    url = f"https://api.reccobeats.com/v1/audio-features?ids={id}"
    payload = {}
    headers = {
      'Accept': 'application/json'
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    features = response.json()['content']
    
    headers = {
      "Authorization": f"Bearer {access_token}"
    }

    df_features = pd.json_normalize(features)
    df_features['id'] = id
    name = requests.get(f"{API_BASE_URL}tracks/{id}", headers=headers).json()
    df_features['name'] = name['name']
    df = pd.concat([df, df_features], ignore_index=True)

  df = clean_audio_features(df)
  df.to_csv("../data/processed_audio_features.csv", index=False)

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