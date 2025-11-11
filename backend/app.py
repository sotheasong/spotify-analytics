import requests
from flask import Flask, redirect, request, jsonify, session, Response, render_template
import os
import urllib.parse
import json
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import io
from flask_cors import CORS

from analysis.cleaning import clean_top_artists, clean_recents, clean_top_tracks
from analysis.analysis import genre_chart

app = Flask(__name__)
app.secret_key = os.urandom(24)

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

REDIRECT_URI = "http://127.0.0.1:5000/callback"

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

  # ids = [t['id'] for t in tracks] 

  artists_res = requests.get(f"{API_BASE_URL}me/top/artists?time_range=long_term&limit=50", headers=headers)
  artists = artists_res.json()['items']

  recent_res = requests.get(f"{API_BASE_URL}me/player/recently-played?limit=50", headers=headers)
  recent = recent_res.json()['items']


  df_tracks = clean_top_tracks(tracks)
  df_artists = clean_top_artists(artists)
  df_recent = clean_recents(recent)

  df_tracks.to_csv("../data/processed_tracks.csv", index=False)
  df_artists.to_csv("../data/processed_artists.csv", index=False)
  df_recent.to_csv("../data/processed_recent.csv", index=False)

  genre_plot = genre_chart(artists)

  return jsonify({
    "message": "Data fetched and processed successfully.",
    "tracks_sample": df_tracks.head().to_dict(orient='records'),
    "artists_sample": df_artists.head().to_dict(orient='records'),
    "recent_sample": df_recent.head().to_dict(orient='records'),
    "genre_plot": genre_plot
  })


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