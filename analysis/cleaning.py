import pandas as pd

def clean_top_tracks(tracks_json):
  """
  Cleans the Spotify top tracks data into a flat, analysis-ready DataFrame.
  """

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