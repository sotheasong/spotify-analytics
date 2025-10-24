import requests
from flask import Flask, redirect, request, jsonify, session
import os
import urllib.parse
import json
import pandas as pd
import numpy as np
from datetime import datetime

from analysis.cleaning import clean_top_tracks




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
  scope = "user-read-private user-read-email user-read-playback-position user-top-read"
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
  
  tracks_res = requests.get(f"{API_BASE_URL}me/top/tracks?limit=50", headers=headers)
  tracks = tracks_res.json()['items']

  ids = [t['id'] for t in tracks]
  # features_res = requests.get(f"{API_BASE_URL}audio-features?ids={','.join(ids)}", headers=headers)

  df_tracks = clean_top_tracks(tracks)
  # df_features = pd.DataFrame(features)
  # df = pd.concat([df_tracks, df_features], axis=1)

  df_tracks.to_csv("data/processed_tracks.csv", index=False)
  
  return jsonify({"message": "Data saved successfully!", "num_tracks": len(df_tracks)})


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
  

if __name__ == "__main__":
  app.run(host="0.0.0.0", debug=True)