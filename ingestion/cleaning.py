import pandas as pd
import requests

API_BASE_URL = "https://api.spotify.com/v1/"

def clean_top_artists(artists_json):
  df = pd.json_normalize(artists_json)

  keep_cols = [
    'id',
    'name',
    'popularity',
    'genres',
    'followers.total'
  ]
  df = df[keep_cols]

  df['genres'] = df['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
  
  df = df.rename(columns={'followers.total': 'follower_count'})
  return df

def clean_recents(recent_json):
  df = pd.json_normalize(recent_json)
  keep_cols = [
      'track.id',
      'track.name',
      'track.artists',
      'track.album.name',
      'played_at'
  ]
  df = df[keep_cols]
  df['artist_name'] = df['track.artists'].apply(
      lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else None
  )
  df = df.drop(columns=['track.artists'])
  df['played_at'] = pd.to_datetime(df['played_at'], errors='coerce')
  df = df.rename(columns={
      'track.id': 'track_id',
      'track.name': 'track_name',
      'track.album.name': 'album_name',
      'played_at': 'played_at'
  })
  return df

def clean_top_tracks(tracks_json):
  # Flatten nested JSON (e.g. album.name -> 'album.name')
  df = pd.json_normalize(tracks_json)

  # Keep only the relevant columns
  keep_cols = [
      'id',
      'name',
      'popularity',
      'duration_ms',
      'explicit',
      'album.name',
      'album.release_date',
      'artists'
  ]
  df = df[keep_cols]

  # Extract the first artist’s name
  df['artist_name'] = df['artists'].apply(
      lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else None
  )

  # Drop the nested artist column now that we’ve extracted the name
  df = df.drop(columns=['artists'])

  # Convert duration from milliseconds to minutes
  df['duration_min'] = df['duration_ms'] / 60000

  # Convert release date to datetime (some are only "YYYY" or "YYYY-MM")
  df['album.release_date'] = pd.to_datetime(
      df['album.release_date'], errors='coerce'
  )

  # Drop rows missing essential fields
  df = df.dropna(subset=['name', 'artist_name'])

  # Reorder columns neatly
  df = df[
      [
          'id',
          'name',
          'artist_name',
          'album.name',
          'album.release_date',
          'duration_min',
          'popularity',
          'explicit'
      ]
  ]

  # Rename for clarity
  df = df.rename(columns={
      'name': 'track_name',
      'album.name': 'album_name',
      'album.release_date': 'release_date'
  })

  return df

def clean_audio_features(features_json):
  df = features_json
  keep_cols = [
      'id',
      'name',
      'acousticness',
      'danceability',
      'energy',
      'instrumentalness',
      'key',
      'liveness',
      'loudness',
      'mode',
      'speechiness',
      'valence',
      'tempo'
  ]
  df = df[keep_cols]
  return df