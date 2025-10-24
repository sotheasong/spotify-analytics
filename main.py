import requests
from flask import Flask, redirect, request, jsonify, session
import os
import urllib.parse
from datetime import datetime

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

  return redirect("/top-artists")


@app.route("/top-artists")
def get_playlists():
  access_token = session.get('access_token')
  if not access_token:
    return redirect("/login")
  
  if datetime.now().timestamp() > session.get('expires_at', 0):
    return redirect("/refresh_token")
  
  headers = {
    "Authorization": f"Bearer {access_token}"
  }
  
  response = requests.get(f"{API_BASE_URL}me/top/{"artists"}", headers=headers)
  top_artists = response.json()
  
  return jsonify(top_artists)


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

    return redirect("/top-artists")
  

if __name__ == "__main__":
  app.run(host="0.0.0.0", debug=True)