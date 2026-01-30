# Goal

To understand the user's listening behavior, and weight out what is more important when assessing what kind of music and artist to recommend to the user.

# Dataset

| File name                          | Description                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| `recent_tracks_audio_features.csv` | User’s recently played tracks with audio features (e.g. acousticness, danceability, energy, etc.) |
| `recent_tracks.csv`                | User’s recently played tracks                                                                     |
| `top_artists.csv`                  | User’s top artists (as of the collection date)                                                    |
| `top_tracks_audio_features.csv`    | User’s top tracks with audio features                                                             |
| `top_tracks.csv`                   | User’s top tracks (as of the collection date)                                                     |


## Audio features

| Feature            | Type  | Description                                                                                  |
| ------------------ | ----- | -------------------------------------------------------------------------------------------- |
| `acousticness`     | Float | Confidence (0.0–1.0) that the track is acoustic. Higher values indicate more natural sounds. |
| `danceability`     | Float | Suitability for dancing (0.0–1.0). Higher values indicate more rhythmically engaging tracks. |
| `energy`           | Float | Intensity and liveliness (0.0–1.0). Higher values indicate more energetic tracks.            |
| `instrumentalness` | Float | Likelihood of no vocals (0.0–1.0). Values above 0.5 suggest instrumental tracks.             |
| `liveness`         | Float | Probability of a live audience (0.0–1.0). Values above 0.8 strongly suggest a live track.    |
| `loudness`         | Float | Average loudness in decibels (dB). Typically ranges between -60 and 0 dB.                    |
| `speechiness`      | Float | Presence of spoken words (0.0–1.0). Values above 0.66 indicate mostly speech.                |
| `tempo`            | Float | Estimated tempo in beats per minute (BPM). Typically ranges between 0 and 250.               |
| `valence`          | Float | Emotional tone (0.0–1.0). Higher values indicate a happier mood; lower values a sadder one.  |


# Questions

What patterns consistently explain why this user listens to what they listen to?

1. Is the user genre-consistent or feature-consistent?
2. Do they explore new artists or replay the same ones?
3. Do preferences change by time (day/night, weekday/weekend)?
4. Do “liked” songs cluster in feature space?

## Weighing candidates

- Audio features (eg. )
- Genre
- Artists

# Database

- Discogs
