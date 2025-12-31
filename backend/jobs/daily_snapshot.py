from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


from backend.app import (
    collect_user_datasets,
    load_persisted_refresh_token,
    persist_snapshot,
    refresh_access_token,
    store_refresh_token,
)

def resolve_refresh_token(args: argparse.Namespace) -> str:
  """Resolve the refresh token from CLI args, env vars, or stored file."""
  if args.refresh_token:
    return args.refresh_token

  env_token = os.getenv("SPOTIFY_REFRESH_TOKEN")
  if env_token:
    return env_token

  stored = load_persisted_refresh_token()
  if stored:
    return stored

  raise RuntimeError(
      "A Spotify refresh token is required. Provide --refresh-token, set "
      "SPOTIFY_REFRESH_TOKEN, or login via the web app once to persist it."
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Capture a Spotify listening snapshot.")
  parser.add_argument(
      "--refresh-token",
      dest="refresh_token",
      help="Spotify refresh token to exchange for an access token",
  )
  args = parser.parse_args()

  load_dotenv()

  refresh_token = resolve_refresh_token(args)

  token_info = refresh_access_token(refresh_token)
  access_token = token_info["access_token"]

  # Persist a newly issued refresh token for future runs.
  store_refresh_token(token_info.get("refresh_token", refresh_token))

  df_tracks, df_artists, df_recent, df_audio_recent, df_audio_top = collect_user_datasets(access_token)
  snapshot_dir = persist_snapshot(
      df_tracks,
      df_artists,
      df_recent,
      df_audio_recent,
      df_audio_top,
      snapshot_time=datetime.now(timezone.utc)
  )

  print(f"Snapshot captured at {snapshot_dir}")


if __name__ == "__main__":
  main()