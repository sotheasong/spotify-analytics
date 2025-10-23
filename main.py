import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

